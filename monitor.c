#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sysinfo.h>
#include <linux/ktime.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/pid.h>
#include <linux/sched.h>
#include <linux/signal.h>

#define MOD_NAME "sysmon"
#define PROC_DIR  "sysmon"
#define PROC_FILE "stats"

static struct proc_dir_entry *proc_dir;
static struct proc_dir_entry *proc_stats;
static struct proc_dir_entry *proc_control;

static int sysmon_stats_show(struct seq_file *m, void *v)
{
    struct sysinfo si;
    u64 timestamp;
    struct task_struct *p;
    struct mm_struct *mm;
    unsigned long top_rss_kb = 0;
    pid_t top_pid = -1;
    int total_tasks = 0;
    int running_tasks = 0;
    
    si_meminfo(&si);
    timestamp = (u64)ktime_get_real_seconds();

    rcu_read_lock();
    for_each_process(p) {
        unsigned long rss_kb = 0;

        mm = get_task_mm(p);
        if (mm) {
            rss_kb = (get_mm_rss(mm) << PAGE_SHIFT) / 1024;
            mmput(mm);
        }

        total_tasks++;
        if (task_is_running(p))
            running_tasks++;

        if (rss_kb > top_rss_kb) {
            top_rss_kb = rss_kb;
            top_pid = p->pid;
        }
    }
    rcu_read_unlock();

    seq_printf(m, "%llu,%lu,%lu,%d,%d,%d,%lu\n",
           timestamp,
           si.totalram * si.mem_unit / 1024,
           si.freeram * si.mem_unit / 1024,
           total_tasks,
           running_tasks,
           top_pid,
           top_rss_kb);
    
    return 0;
}

static int sysmon_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, sysmon_stats_show, NULL);
}

static const struct proc_ops sysmon_stats_proc_ops = {
    .proc_open    = sysmon_stats_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};

static ssize_t sysmon_control_write(struct file *file, const char __user *buf,
                    size_t len, loff_t *ppos)
{
    char kbuf[32];
    long pid_val;
    struct pid *pid_struct;

    if (len == 0 || len >= sizeof(kbuf))
        return -EINVAL;

    if (copy_from_user(kbuf, buf, len))
        return -EFAULT;

    kbuf[len] = '\0';

    if (kstrtol(kbuf, 10, &pid_val))
        return -EINVAL;

    if (pid_val <= 1)
        return -EINVAL;

    rcu_read_lock();
    pid_struct = find_vpid(pid_val);
    if (pid_struct)
        kill_pid(pid_struct, SIGKILL, 1);
    rcu_read_unlock();

    return len;
}

static const struct proc_ops sysmon_control_proc_ops = {
    .proc_write = sysmon_control_write,
};

static int __init sysmon_init(void)
{
    proc_dir = proc_mkdir(PROC_DIR, NULL);
    if (!proc_dir) {
        pr_err(MOD_NAME ": failed to create proc dir\n");
        return -ENOMEM;
    }
    
    proc_stats = proc_create(PROC_FILE, 0444, proc_dir, &sysmon_stats_proc_ops);
    if (!proc_stats) {
        proc_remove(proc_dir);
        pr_err(MOD_NAME ": failed to create proc file\n");
        return -ENOMEM;
    }

    proc_control = proc_create("control", 0222, proc_dir, &sysmon_control_proc_ops);
    if (!proc_control) {
        proc_remove(proc_stats);
        proc_remove(proc_dir);
        pr_err(MOD_NAME ": failed to create control file\n");
        return -ENOMEM;
    }
    
    pr_info(MOD_NAME ": module loaded, stats at /proc/%s/%s\n", PROC_DIR, PROC_FILE);
    return 0;
}

static void __exit sysmon_exit(void)
{
    if (proc_control)
        proc_remove(proc_control);
    if (proc_stats)
        proc_remove(proc_stats);
    if (proc_dir)
        proc_remove(proc_dir);
    pr_info(MOD_NAME ": module unloaded\n");
}

module_init(sysmon_init);
module_exit(sysmon_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("System monitor kernel module");

