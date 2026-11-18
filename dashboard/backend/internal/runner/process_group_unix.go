//go:build !windows

package runner

import (
	"os/exec"
	"syscall"
	"time"
)

func configureProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminateProcess(cmd *exec.Cmd, force bool) error {
	if cmd.Process == nil {
		return nil
	}
	if force {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM); err != nil {
		return err
	}
	time.Sleep(150 * time.Millisecond)
	return nil
}
