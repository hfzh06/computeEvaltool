package stub

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/ict/tide/pkg/clsstat"
	"github.com/ict/tide/pkg/dev"
	"github.com/ict/tide/pkg/interfaces"
	"github.com/ict/tide/pkg/processor"
	"github.com/ict/tide/pkg/routine"
	"github.com/sirupsen/logrus"
)

func (s *stub) processFnLoop() {
	logrus.WithFields(logrus.Fields{
		"StubId": s.id,
	}).Info("Stub: enter process loop")
	s.status = RUNNING

	for {
		select {
		case <-s.done:
			return

		case r := <-s.rtnStore.Pop():
			s.processOne(r)
		}
	}
}

func (s *stub) processOne(r *routine.Routine) {
	logger := logrus.WithFields(logrus.Fields{
		"StubId":    s.id,
		"RoutineId": r.Id,
	})
	logger.Debug("Stub: start to process this routine, and acquire a proc")

	proc := s.acquireProcessor(r)
	if proc == nil { // stub is closed
		return
	}
	// logger.WithField("ProcessorId", proc.Id).Debug("Stub: got a processor")
	logger.WithFields(logrus.Fields{
		"ProcessorId":      proc.Id,
		"logic resource":   proc.LogicRes,
		"physics resource": proc.PhysicRes,
	}).Debug("Stub: got a processor")
	go s.executeRoutine(proc, r)
}

func (s *stub) acquireProcessor(r *routine.Routine) *processor.Processor {
	// EXP 如果不并发控制，则直接使用高优资源
	if dev.RSEPTest && !dev.EnableConcurrencyControl {
		return s.bindProc
	}

	select {
	case <-s.done:
		return nil

	case <-s.bindCh:
		return s.bindProc

	default: // 先尝试bindCh，没有再尝试ProcM抢用
		ctx, cancel := context.WithCancel(context.Background())
		ctx = context.WithValue(ctx, clsstat.ApplicationId("appid"), s.appId)
		defer cancel()
		select {
		case <-s.done:
			return nil

		case <-s.bindCh:
			return s.bindProc

		case proc := <-s.procM.Rob(ctx, r.ResReq):
			return proc
		}
	}
}

func (s *stub) executeRoutine(proc *processor.Processor, rtn *routine.Routine) {
	defer atomic.AddInt64(&s.noExecuting, -1) // 注意 defer 执行顺序是反的，需先reset
	defer s.timer.Reset(s.timerGap)

	logger := logrus.WithFields(logrus.Fields{
		"StubId":      s.id,
		"ProcessorId": proc.Id,
		"RoutineId":   rtn.Id,
	})

	logger.Debug("Stub: try to get executor")
	exec := s.execPool.Get()
	if exec == nil { // executor pool exitd, the stub also exited
		return
	}
	logger = logger.WithField("ExecutorId", exec.Id())
	logger.Debug("Stub: got a executor")

	// TODO: need handle the routine
	if err := exec.BindProc(proc); err != nil {
		logrus.WithFields(logrus.Fields{
			"StubId":     s.id,
			"ExecutorId": exec.Id(),
		}).Error("Stub: failed to bind processor to executor, ", err)
		goto onerr
	}
	if err := exec.BindRtn(rtn); err != nil {
		logrus.WithFields(logrus.Fields{
			"StubId":     s.id,
			"ExecutorId": exec.Id(),
		}).Error("Stub: failed to bind routine to executor, ", err)
		goto onerr
	}

	s.mu.Lock()
	if proc == s.bindProc {
		s.rtnOnBindProc = rtn
		s.startTime = time.Now()
	}
	// 队列长度减小，队列中等待执行执行的预估等待时间发生改变
	s.accExecTimeInQue -= int64(dev.GetEstExecTime(clsstat.Singleton().GetSelf().Id))
	go s.noticeStubUpdate(s.estWaitTime(), time.Now())
	s.mu.Unlock()

	logger.Debug("Stub: notice executor to execute")
	if err := exec.Execute(); err != nil {
		s.postProcess(exec)
		logrus.Fatal(err)
	}
	s.listenEvt(exec)
	s.postProcess(exec)
	return

onerr:
	s.returnProc(proc)
}

func (s *stub) postProcess(exec interfaces.Executor) error {
	logger := logrus.WithFields(logrus.Fields{
		"StubId":     s.id,
		"ExecutorId": exec.Id(),
	})
	logger.Debug("Stub: start post process")

	// unbind routine
	rtn, err := exec.UnbindRtn()
	if err != nil {
		logger.Fatal("Stub: failed to unbind routine, ", err)
	}
	logger = logger.WithField("RoutineId", rtn.Id)
	logger.Debugf("Stub: finish processing one routine, %v", rtn)

	defer logger.Debug("Stub: finish post process")

	// unbind processor
	proc, err := exec.UnbindProc()
	if err != nil {
		logger.Error("Stub: failed to unbind processor, ", err)
		return err
	}

	// 必须先还Executor，再还Processor
	// 如果先还Processor，Stub获取后又申请新的Executor，而这里有一个空闲的Executor尚未归还，导致创建新的Executor
	// 但是会导致Proc有一个空闲时间
	s.execPool.Return(exec)
	s.returnProc(proc)
	return nil
}

func (s *stub) returnProc(proc *processor.Processor) {
	// EXP
	if dev.RSEPTest && !dev.EnableConcurrencyControl {
		return
	}

	if s.bindProc != proc {
		s.procM.Return(proc, true)

	} else {
		s.mu.Lock()
		s.startTime = time.Time{}
		s.rtnOnBindProc = nil
		go s.noticeStubUpdate(s.estWaitTime(), time.Now())
		s.mu.Unlock()

		s.bindCh <- struct{}{}
	}
}
