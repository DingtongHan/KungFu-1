package monitorserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"net/http"
	//"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type monitorServer struct {
	DownFlag        bool
	Machines        int
	Epochnum        int
	FinishFlag      bool
	trainend        []int
	times           []int64
	epochs          []int
	wg              sync.WaitGroup
	OtherFinish     bool
	OtherEpochnum   int
	OtherDown       bool
	serverip        string
	NewHL           plan.HostList
	Newcluster      plan.Cluster
	NewClusterSize  int
	MachineDownFlag bool
	LocalHost       uint32
}

type Results struct {
	DownFlag       bool
	Epochnum       int
	FinishFlag     bool
	NewClusterFlag bool
	NewCluster     plan.Cluster
	NewClusterSize int
	NewHL          plan.HostList
}

type Message struct {
	Key string `json:"key"`
	HL  plan.HostList
	CS  int
}

func (h *monitorServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var msg Message
	err := json.NewDecoder(r.Body).Decode(&msg)
	if err != nil {
		return
	}
	datas := strings.Split(string(msg.Key), ":")
	intva, err := strconv.Atoi(datas[1])
	if err != nil {
	}
	if datas[0] == "trainend" {
		h.trainend[intva] = 1
	}
	if datas[0] == "begin" {
		h.times[intva] = time.Now().Unix()
	}
	if datas[0] == "end" {
		h.times[intva] = 0
	}
	if datas[0] == "epoch" {
		h.epochs[intva] = h.epochs[intva] + 1
	}
	if datas[0] == "otherfinish" {
		h.OtherFinish = true
	}
	if datas[0] == "otherdown" {
		h.OtherEpochnum = intva
		h.OtherDown = true
	}
	if datas[0] == "othermachinedown" {
		h.OtherEpochnum = intva
		h.OtherDown = true
		h.MachineDownFlag = true
		h.NewHL = msg.HL
		h.NewClusterSize = msg.CS
		var peers plan.PeerList
		var runners plan.PeerList
		runners = h.NewHL.GenRunnerList(uint16(int(plan.DefaultRunnerPort))) // FIXME: assuming runner port is the same
		self := plan.PeerID{IPv4: h.LocalHost, Port: uint16(int(plan.DefaultRunnerPort))}
		if _, ok := runners.Rank(self); !ok {
			utils.ExitErr(fmt.Errorf("%s not in %s", self, runners))
		}
		peers, err = h.NewHL.GenPeerList(h.NewClusterSize, plan.DefaultPortRange)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
		}
		h.Newcluster = plan.Cluster{
			Runners: runners,
			Workers: peers,
		}
	}
}

func (s *monitorServer) Start(ipv4 uint32, hl plan.HostList, clusterSize int, waitTime time.Duration) {
	s.LocalHost = ipv4
	var otherips []string
	ip := plan.FormatIPv4(ipv4)
	isMainServer := false
	ipnum := 0
	if len(hl) != 1 {
		for _, h := range hl {
			if plan.FormatIPv4(h.IPv4) == ip {
				if ipnum == 0 {
					isMainServer = true
				}
			} else {
				otherips = append(otherips, plan.FormatIPv4(h.IPv4))
			}
			ipnum = ipnum + 1
		}
	}
	server := &http.Server{
		Addr:    ip + ":7756",
		Handler: s,
	}
	defer s.wg.Done()
	for i := 0; i < clusterSize; i++ {
		s.trainend = append(s.trainend, 0)
		s.times = append(s.times, 0)
		s.epochs = append(s.epochs, 0)
	}
	go func() {
		server.ListenAndServe()
		s.wg.Done()
	}()
	for {
		time.Sleep(1)
		trainendflag := 0
		downflag := false
		for i := 0; i < clusterSize; i++ {
			if s.trainend[i] == 1 {
				trainendflag = trainendflag + 1
			}
			if a := time.Duration(time.Now().Unix()-s.times[i]) * time.Second; a > waitTime && s.times[i] != 0 {
				min := findmin(s.epochs)
				s.DownFlag = true
				s.Epochnum = min
				downflag = true
				ismachinedown := false
				if len(hl) != 1 {
					var newotherips []string
					var newHS []plan.HostSpec
					var newHl plan.HostList
					newipnum := 0
					for _, h := range hl {
						if plan.FormatIPv4(h.IPv4) != ip {
                                                        contentType := "application/json;charset=utf-8"
							data := "judge:0"
							msg := Message{Key: data}
							b, err := json.Marshal(msg)
							if err != nil {
								return
							}
							body := bytes.NewBuffer(b)
                                                        Client := http.Client{
                                                                Timeout:3*time.Second,
                                                        }
							resp, err := Client.Post("http://"+plan.FormatIPv4(h.IPv4)+":7756", contentType, body)
							if err != nil {
								ismachinedown = true
								clusterSize = clusterSize - 1
							} else {
								newHS = append(newHS, h)
                                                                newotherips = append(newotherips, plan.FormatIPv4(h.IPv4))
								defer resp.Body.Close()
							}
							
						} else {
							newHS = append(newHS, h)

						}
                                                newipnum = newipnum + 1
					}
                                        s.NewClusterSize = clusterSize
					newHl = newHS
					if ismachinedown {
						s.MachineDownFlag = true
						s.NewHL = newHl
						var peers plan.PeerList
						var runners plan.PeerList
						runners = newHl.GenRunnerList(uint16(int(plan.DefaultRunnerPort))) // FIXME: assuming runner port is the same
						self := plan.PeerID{IPv4: ipv4, Port: uint16(int(plan.DefaultRunnerPort))}
						if _, ok := runners.Rank(self); !ok {
							utils.ExitErr(fmt.Errorf("%s not in %s", self, runners))
						}
						peers, _ = newHl.GenPeerList(clusterSize, plan.DefaultPortRange)
						//if err != nil {
						//utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
						//}
						s.Newcluster = plan.Cluster{
							Runners: runners,
							Workers: peers,
						}
						contentType := "application/json;charset=utf-8"
						data := "othermachinedown:" + strconv.Itoa(min)
                                                s.Epochnum = min
						msg := Message{Key: data, HL: s.NewHL, CS: clusterSize}
						b, err := json.Marshal(msg)
						if err != nil {
							return
						}
						body := bytes.NewBuffer(b)
						for _, otherip := range newotherips {
							resp, err := http.Post("http://"+otherip+":7756", contentType, body)
							if err != nil {
								return
							}
							defer resp.Body.Close()
						}
					}
					break

				}
				if isMainServer {
					contentType := "application/json;charset=utf-8"
					data := "otherdown:" + strconv.Itoa(min)
                                        s.Epochnum = min
					msg := Message{Key: data}
					b, err := json.Marshal(msg)
					if err != nil {
						return
					}
					body := bytes.NewBuffer(b)
					for _, otherip := range otherips {
						resp, err := http.Post("http://"+otherip+":7756", contentType, body)
						if err != nil {
							return
						}
						defer resp.Body.Close()
					}
				}
				break
			}
		}
		if s.OtherDown || s.MachineDownFlag {
			s.DownFlag = true
                        if s.Epochnum == 0  {
                            s.Epochnum = s.OtherEpochnum
                        }
			break
		}
		if downflag {
			break
		}

		if trainendflag == clusterSize || s.OtherFinish {
			if isMainServer {
				contentType := "application/json;charset=utf-8"
				data := "otherfinish:0"
				msg := Message{Key: data}
				b, err := json.Marshal(msg)
				if err != nil {
					return
				}
				body := bytes.NewBuffer(b)
				for _, otherip := range otherips {
					resp, err := http.Post("http://"+otherip+":7756", contentType, body)
					if err != nil {
						fmt.Println(err)

						return
					}
					defer resp.Body.Close()
				}
			}
			s.FinishFlag = true
			break
		}
	}
	server.Shutdown(context.TODO())
}

func findmin(array []int) int {
	min := array[0]
	for _, v := range array {
		if v < min {
			min = v
		}
	}
	return min
}

func (s *monitorServer) Wait() Results {
	s.wg.Wait()
	return Results{
		DownFlag:       s.DownFlag,
		Epochnum:       s.Epochnum,
		FinishFlag:     s.FinishFlag,
		NewClusterFlag: s.MachineDownFlag,
		NewCluster:     s.Newcluster,
		NewClusterSize: s.NewClusterSize,
		NewHL:          s.NewHL,
	}
}

func New(procs int) *monitorServer {
	return &monitorServer{Machines: procs}
}

func (s *monitorServer) Monitor(ip uint32, hl plan.HostList, clusterSize int, waitTime time.Duration) {
	s.wg.Add(2)
	go s.Start(ip, hl, clusterSize, waitTime)
}
