package runner

import (
	"context"
        "sync/atomic"
        "sync"
	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

func MonitoredRun(ctx context.Context, selfIPv4 uint32, cluster plan.Cluster, j job.Job, verboseLog bool) {
        for{
                ctx, cancel := context.WithCancel(ctx)
	        defer cancel()
                var failfi int32
                var sucessfi int32
                var cont int32
		procs := j.CreateProcs(cluster, selfIPv4)
                s := New(len(procs))
                
		log.Infof("will parallel run %d instances of %s with %q under monitor", len(procs), j.Prog, j.Args)
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			d, err := utils.Measure(func() error { return local.RunAll(ctx, procs, verboseLog) })
			log.Infof("all %d/%d local peers finished, took %s", len(procs), len(cluster.Workers), d)
			if err != nil {
                            utils.ExitErr(err)
			    atomic.AddInt32(&failfi, 1)
			} else{
                            atomic.AddInt32(&sucessfi, 1)
                        }
			wg.Done()
		}()
	        s.Start()
		Results := s.Wait()
		
                if Results.FinishFlag == 1 {
                        atomic.AddInt32(&sucessfi, 1)
                }
                if Results.DownFlag == 1 {
                        atomic.AddInt32(&cont, 1)
                }
                if failfi == 1 {
                    log.Infof("fail finish")
                    break
	        }
                if sucessfi == 2 {
                    log.Infof("sucessc finish")
                    break
	        }
                if cont != 0 {
                    log.Infof("server down")
                    continue
	        }
                wg.Wait()
        }
}
