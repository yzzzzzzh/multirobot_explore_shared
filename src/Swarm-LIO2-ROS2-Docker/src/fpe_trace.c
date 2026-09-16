/* LD_PRELOAD SIGFPE tracer: on a floating-point exception, record si_code
 * (FPE_INTDIV=1 => integer divide-by-zero, FPE_FLTDIV=3 => float) and a
 * backtrace, then re-raise with the default handler so the process still dies.
 * Build: gcc -shared -fPIC -O0 -g fpe_trace.c -o fpe_trace.so -ldl -rdynamic
 */
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

static void handler(int sig, siginfo_t *si, void *ctx) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/fpe_bt_%d.txt", getpid());
    FILE *f = fopen(path, "w");
    if (f) {
        const char *nm = sig == SIGFPE ? "SIGFPE" : sig == SIGSEGV ? "SIGSEGV"
                       : sig == SIGABRT ? "SIGABRT" : "SIG?";
        fprintf(f, "%s pid=%d si_code=%d addr=%p\n",
                nm, getpid(), si->si_code, si->si_addr);
        void *bt[64];
        int n = backtrace(bt, 64);
        fflush(f);
        backtrace_symbols_fd(bt, n, fileno(f));
        fclose(f);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor))
static void install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}
