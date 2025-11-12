#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"

// Added for CSC3150 HW3: multi-level feedback queue scheduler state.
#define MLFQ_LEVELS 3
static const int mlfq_slices[MLFQ_LEVELS] = {3, 6, 12};
#define MLFQ_BOOST_INTERVAL 100

static uint64 mlfq_enqueue_counter = 0;
static uint64 mlfq_tick_counter = 0;
static volatile int mlfq_need_boost = 0;

static struct {
  struct spinlock lock;
  uint64 generation;
  uint64 total_response_ticks;
  uint64 completed_processes;
  uint64 first_creation_tick;
  uint64 last_completion_tick;
} sched_stats;

static uint64 next_mlfq_stamp(void);
static void mlfq_requeue_locked(struct proc *p);
static void mlfq_try_boost(void);
static void mlfq_apply_boost(void);
static uint64 read_sched_ticks(void);
static void schedstats_record_creation(struct proc *p);
static void schedstats_record_first_run(struct proc *p, uint64 now);
static void schedstats_record_completion(struct proc *p, uint64 now);

struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

static uint64
next_mlfq_stamp(void)
{
  return __sync_fetch_and_add(&mlfq_enqueue_counter, 1);
}

static uint64
read_sched_ticks(void)
{
  return __sync_fetch_and_add(&mlfq_tick_counter, 0);
}

static void
mlfq_requeue_locked(struct proc *p)
{
  p->ticks_in_level = 0;
  p->queue_stamp = next_mlfq_stamp();
}

static void
mlfq_apply_boost(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state != UNUSED) {
      int old_level = p->queue_level;
      p->queue_level = 0;
      p->ticks_in_level = 0;
      if(p->state == RUNNABLE) {
        if(old_level != 0) {
          printf("process %d promoted to level %d\n", p->pid, p->queue_level);
        }
        p->queue_stamp = next_mlfq_stamp();
      }
    }
    release(&p->lock);
  }
}

static void
schedstats_record_creation(struct proc *p)
{
  uint64 now = read_sched_ticks();

  acquire(&sched_stats.lock);
  p->creation_tick = now;
  p->first_run_tick = 0;
  p->completion_tick = 0;
  p->stats_generation = sched_stats.generation;
  if(sched_stats.first_creation_tick == 0 ||
     now < sched_stats.first_creation_tick) {
    sched_stats.first_creation_tick = now;
  }
  release(&sched_stats.lock);
}

static void
schedstats_record_first_run(struct proc *p, uint64 now)
{
  if(p->first_run_tick != 0)
    return;

  p->first_run_tick = now;

  acquire(&sched_stats.lock);
  if(p->stats_generation == sched_stats.generation &&
     now >= p->creation_tick) {
    sched_stats.total_response_ticks += now - p->creation_tick;
  }
  release(&sched_stats.lock);
}

static void
schedstats_record_completion(struct proc *p, uint64 now)
{
  p->completion_tick = now;

  acquire(&sched_stats.lock);
  if(p->stats_generation == sched_stats.generation) {
    if(sched_stats.last_completion_tick == 0 ||
       now > sched_stats.last_completion_tick) {
      sched_stats.last_completion_tick = now;
    }
    if(sched_stats.first_creation_tick == 0 ||
       p->creation_tick < sched_stats.first_creation_tick) {
      sched_stats.first_creation_tick = p->creation_tick;
    }
    sched_stats.completed_processes++;
  }
  release(&sched_stats.lock);
}

static void
mlfq_try_boost(void)
{
  if(__sync_lock_test_and_set(&mlfq_need_boost, 0)) {
    mlfq_apply_boost();
  }
}

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table.
void
procinit(void)
{
  struct proc *p;

  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  initlock(&sched_stats.lock, "schedstats");
  sched_stats.generation = 1;
  sched_stats.total_response_ticks = 0;
  sched_stats.completed_processes = 0;
  sched_stats.first_creation_tick = 0;
  sched_stats.last_completion_tick = 0;
  for(p = proc; p < &proc[NPROC]; p++) {
      initlock(&p->lock, "proc");
      p->state = UNUSED;
      p->kstack = KSTACK((int) (p - proc));
  }
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int
cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu*
mycpu(void)
{
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc*
myproc(void)
{
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int
allocpid()
{
  int pid;
  
  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  // Initialize MLFQ bookkeeping for new processes.
  p->queue_level = 0;
  p->ticks_in_level = 0;
  p->queue_stamp = next_mlfq_stamp();
  schedstats_record_creation(p);

  return p;
}

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
  p->queue_level = 0;
  p->ticks_in_level = 0;
  p->queue_stamp = 0;
  p->creation_tick = 0;
  p->first_run_tick = 0;
  p->completion_tick = 0;
  p->stats_generation = 0;
}

// Create a user page table for a given process, with no user memory,
// but with trampoline and trapframe pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

// Free a process's page table, and free the
// physical memory it refers to.
void
proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}

// Set up first user process.
void
userinit(void)
{
  struct proc *p;

  p = allocproc();
  initproc = p;
  
  p->cwd = namei("/");

  p->state = RUNNABLE;
  mlfq_requeue_locked(p);

  release(&p->lock);
}

// Grow or shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint64 sz;
  struct proc *p = myproc();

  sz = p->sz;
  if(n > 0){
    if(sz + n > TRAPFRAME) {
      return -1;
    }
    if((sz = uvmalloc(p->pagetable, sz, sz + n, PTE_W)) == 0) {
      return -1;
    }
  } else if(n < 0){
    sz = uvmdealloc(p->pagetable, sz, sz + n);
  }
  p->sz = sz;
  return 0;
}

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int
kfork(void)
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // Allocate process.
  if((np = allocproc()) == 0){
    return -1;
  }

  // Copy user memory from parent to child.
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  pid = np->pid;

  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  mlfq_requeue_locked(np);
  release(&np->lock);

  return pid;
}

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void
reparent(struct proc *p)
{
  struct proc *pp;

  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp->parent == p){
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void
kexit(int status)
{
  struct proc *p = myproc();

  if(p == initproc)
    panic("init exiting");

  uint64 now = read_sched_ticks();
  schedstats_record_completion(p, now);

  // Close all open files.
  for(int fd = 0; fd < NOFILE; fd++){
    if(p->ofile[fd]){
      struct file *f = p->ofile[fd];
      fileclose(f);
      p->ofile[fd] = 0;
    }
  }

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;

  release(&wait_lock);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
kwait(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    // Scan through table looking for exited children.
    havekids = 0;
    for(pp = proc; pp < &proc[NPROC]; pp++){
      if(pp->parent == p){
        // make sure the child isn't still in exit() or swtch().
        acquire(&pp->lock);

        havekids = 1;
        if(pp->state == ZOMBIE){
          // Found one.
          pid = pp->pid;
          if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate,
                                  sizeof(pp->xstate)) < 0) {
            release(&pp->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(pp);
          release(&pp->lock);
          release(&wait_lock);
          return pid;
        }
        release(&pp->lock);
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || killed(p)){
      release(&wait_lock);
      return -1;
    }
    
    // Wait for a child to exit.
    sleep(p, &wait_lock);  //DOC: wait-sleep
  }
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void)
{
  struct proc *p;
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;){
    // The most recent process to run may have had interrupts
    // turned off; enable them to avoid a deadlock if all
    // processes are waiting. Then turn them back off
    // to avoid a possible race between an interrupt
    // and wfi.
    intr_on();
    intr_off();

    // MLFQ-specific periodic priority boost hook.
    mlfq_try_boost();

    struct proc *chosen = 0;
    int best_level = MLFQ_LEVELS;
    uint64 best_stamp = 0;

    // Select the runnable process from the highest-priority MLFQ queue.
    for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state == RUNNABLE) {
        int level = p->queue_level;
        uint64 stamp = p->queue_stamp;
        if(chosen == 0 || level < best_level ||
           (level == best_level && stamp < best_stamp)) {
          chosen = p;
          best_level = level;
          best_stamp = stamp;
        }
      }
      release(&p->lock);
    }

    if(chosen) {
      acquire(&chosen->lock);
      if(chosen->state == RUNNABLE) {
        uint64 now = read_sched_ticks();
        schedstats_record_first_run(chosen, now);
        chosen->state = RUNNING;
        c->proc = chosen;
        swtch(&c->context, &chosen->context);

        // Process is done running for now.
        // It should have changed its p->state before coming back.
        c->proc = 0;
      }
      release(&chosen->lock);
      continue;
    }

    // nothing to run; stop running on this core until an interrupt.
    asm volatile("wfi");
  }
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched RUNNING");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  mlfq_requeue_locked(p);
  sched();
  release(&p->lock);
}

int
sched_tick(void)
{
  int yield_now = 0;
  struct proc *p = myproc();
  uint64 current_time;

  // Added to drive MLFQ accounting and periodic boosts from the timer.
  if(cpuid() == 0) {
    current_time = __sync_add_and_fetch(&mlfq_tick_counter, 1);
    if(current_time % MLFQ_BOOST_INTERVAL == 0) {
      __sync_lock_test_and_set(&mlfq_need_boost, 1);
      if(p != 0)
        yield_now = 1;
    }
  } else {
    current_time = __sync_fetch_and_add(&mlfq_tick_counter, 0);
  }

  if(p == 0)
    return yield_now;

  if(mlfq_need_boost)
    yield_now = 1;

  acquire(&p->lock);
  if(p->state == RUNNING) {
    int queue = p->queue_level;
    int slice = mlfq_slices[queue];

    p->ticks_in_level++;

    int remaining = slice - p->ticks_in_level;
    if(remaining < 0)
      remaining = 0;

    printf("PID %d ran at t = %lu in Q%d, remaining ticks = %d.\n",
           p->pid, current_time, queue, remaining);

    if(p->ticks_in_level >= slice) {
      if(p->queue_level < MLFQ_LEVELS - 1) {
        int old_level = p->queue_level;
        p->queue_level++;
        if(p->queue_level != old_level) {
          printf("process %d demoted to level %d\n", p->pid, p->queue_level);
        }
      }
      p->ticks_in_level = 0;
      p->queue_stamp = next_mlfq_stamp();
      yield_now = 1;
    }
  }
  release(&p->lock);
  return yield_now;
}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void
forkret(void)
{
  extern char userret[];
  static int first = 1;
  struct proc *p = myproc();

  // Still holding p->lock from scheduler.
  release(&p->lock);

  if (first) {
    // File system initialization must be run in the context of a
    // regular process (e.g., because it calls sleep), and thus cannot
    // be run from main().
    fsinit(ROOTDEV);

    first = 0;
    // ensure other cores see first=0.
    __sync_synchronize();

    // We can invoke kexec() now that file system is initialized.
    // Put the return value (argc) of kexec into a0.
    p->trapframe->a0 = kexec("/init", (char *[]){ "/init", 0 });
    if (p->trapframe->a0 == -1) {
      panic("exec");
    }
  }

  // return to user space, mimicing usertrap()'s return.
  prepare_return();
  uint64 satp = MAKE_SATP(p->pagetable);
  uint64 trampoline_userret = TRAMPOLINE + (userret - trampoline);
  ((void (*)(uint64))trampoline_userret)(satp);
}

void
schedstats_reset(void)
{
  acquire(&sched_stats.lock);
  sched_stats.generation++;
  sched_stats.total_response_ticks = 0;
  sched_stats.completed_processes = 0;
  sched_stats.first_creation_tick = 0;
  sched_stats.last_completion_tick = 0;
  release(&sched_stats.lock);
}

void
schedstats_report(void)
{
  uint64 generation;
  uint64 total_response;
  uint64 completed;
  uint64 first_tick;
  uint64 last_tick;

  acquire(&sched_stats.lock);
  generation = sched_stats.generation;
  total_response = sched_stats.total_response_ticks;
  completed = sched_stats.completed_processes;
  first_tick = sched_stats.first_creation_tick;
  last_tick = sched_stats.last_completion_tick;
  release(&sched_stats.lock);

  if(completed == 0) {
    printf("[SCHED] Generation %lu: no completed processes to report.\n",
           (unsigned long)generation);
    return;
  }

  uint64 avg_response = total_response / completed;
  uint64 duration = (last_tick > first_tick) ? (last_tick - first_tick) : 0;

  printf("[SCHED] Generation %lu summary:\n", (unsigned long)generation);
  printf("[SCHED]   Completed processes: %lu\n", (unsigned long)completed);
  printf("[SCHED]   Average response time: %lu ticks\n",
         (unsigned long)avg_response);

  if(duration == 0) {
    printf("[SCHED]   Throughput: %lu processes (duration < 1 tick)\n",
           (unsigned long)completed);
    return;
  }

  uint64 throughput_milli = (completed * 1000) / duration;

  printf("[SCHED]   Throughput: %lu.%03lu processes per tick (%lu ticks span)\n",
         (unsigned long)(throughput_milli / 1000),
         (unsigned long)(throughput_milli % 1000),
         (unsigned long)duration);
}

// Sleep on channel chan, releasing condition lock lk.
// Re-acquires lk when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();
  
  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock);  //DOC: sleeplock1
  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}

// Wake up all processes sleeping on channel chan.
// Caller should hold the condition lock.
void
wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
        mlfq_requeue_locked(p);
      }
      release(&p->lock);
    }
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int
kkill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
        mlfq_requeue_locked(p);
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

void
setkilled(struct proc *p)
{
  acquire(&p->lock);
  p->killed = 1;
  release(&p->lock);
}

int
killed(struct proc *p)
{
  int k;
  
  acquire(&p->lock);
  k = p->killed;
  release(&p->lock);
  return k;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int
either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if(user_dst){
    return copyout(p->pagetable, dst, src, len);
  } else {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int
either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if(user_src){
    return copyin(p->pagetable, dst, src, len);
  } else {
    memmove(dst, (char*)src, len);
    return 0;
  }
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}
