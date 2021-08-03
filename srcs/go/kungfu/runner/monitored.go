package runner

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/runner/monitorserver"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

func MonitoredRun(ctx context.Context, selfIPv4 uint32, cluster plan.Cluster, j job.Job, verboseLog bool, hostList plan.HostList, clusterSize int, waitTime time.Duration) {
	t1 := time.Now()
	for {
		t1 = time.Now()
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		var successFinished int32
		var cont int32 // continue flag
		procs := j.CreateProcs(cluster, selfIPv4)
		s := monitorserver.New(0)
		log.Infof("will parallel run %d instances of %s with %q under monitor", len(procs), j.Prog, j.Args)
		s.Monitor(selfIPv4, hostList, clusterSize, waitTime)
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			log.Infof("took %s", time.Since(t1))
			d, err := utils.Measure(func() error { return local.RunAll(ctx, procs, verboseLog) })
			log.Infof("all %d/%d local peers finished, took %s", len(procs), len(cluster.Workers), d)
			if err != nil {
				if cont == 1 {

				} else {
					utils.ExitErr(err)
				}
			} else {
				atomic.AddInt32(&successFinished, 1)
			}
			wg.Done()
		}()

		Results := s.Wait()
		if Results.FinishFlag {
			atomic.AddInt32(&successFinished, 1)
		}
		if Results.DownFlag {
			t1 = time.Now()
			atomic.AddInt32(&cont, 1)
			cancel()
			log.Infof("some machine down")
			for key, value := range j.Args {
				if value == "--n-epochs" || value == "--num-epochs" {
					epochini, err := strconv.Atoi(j.Args[key+1])
					if err != nil {
					}
					j.Args[key+1] = strconv.Itoa(epochini - Results.Epochnum)
				}
			}
			j.Args = append(j.Args, "--restart")
			j.Args = append(j.Args, "1")
                        if Results.NewClusterFlag{
                            cluster = Results.NewCluster
                            clusterSize = Results.NewClusterSize
                            hostList = Results.NewHL
                        }
			continue
		}
		wg.Wait()
		if successFinished == 2 {
			log.Infof("success finish")
			break
		}
	}
}
