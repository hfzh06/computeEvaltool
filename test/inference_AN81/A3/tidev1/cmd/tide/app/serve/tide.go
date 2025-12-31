package serve

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ict/tide/pkg/clsstat"
	"github.com/ict/tide/pkg/cmnct"
	"github.com/ict/tide/pkg/containerx"
	"github.com/ict/tide/pkg/dev"
	"github.com/ict/tide/pkg/event"
	"github.com/ict/tide/pkg/interfaces"
	"github.com/ict/tide/pkg/procmgr"
	"github.com/ict/tide/pkg/routine"
	"github.com/ict/tide/pkg/server"
	"github.com/ict/tide/pkg/server/handlers/cloudhdlr"
	"github.com/ict/tide/pkg/server/handlers/clsstathdlr"
	"github.com/ict/tide/pkg/server/handlers/stubmgrhdlr"
	"github.com/ict/tide/pkg/stringx"
	"github.com/ict/tide/pkg/stubmgr"
	"github.com/ict/tide/proto/commpb"
	"github.com/ict/tide/proto/eventpb"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"golang.org/x/sys/unix"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

var ServeCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start tide serve process",
	Long:  `Start tide serve process.`,
	Run: func(cmd *cobra.Command, args []string) {
		serveTide()
	},
}

var (
	host      string
	port      int
	cpuNum    int
	conc      int
	addresses string
	nodeName  string
	roleParam string
)

func init() {
	ServeCmd.Flags().StringVar(&host, "host", getHost(), "the outbound ip")
	ServeCmd.Flags().IntVarP(&port, "port", "p", 10001, "the server port")
	ServeCmd.Flags().IntVarP(&conc, "conc", "c", 1, "the max concurrent routines")
	ServeCmd.Flags().StringVarP(&addresses, "addresses", "a", "", "the other servers")
	ServeCmd.Flags().StringVar(&roleParam, "role", "things", "the role of this device")
}

func initEnv() {
	var rLim unix.Rlimit
	if err := unix.Getrlimit(unix.RLIMIT_MSGQUEUE, &rLim); err != nil {
		panic(err)
	}
	logrus.Info("Rlimit init: ", rLim)
	if err := unix.Setrlimit(unix.RLIMIT_MSGQUEUE, &unix.Rlimit{Cur: unix.RLIM_INFINITY, Max: unix.RLIM_INFINITY}); err != nil {
		panic(err)
	}
	if err := unix.Getrlimit(unix.RLIMIT_MSGQUEUE, &rLim); err != nil {
		panic(err)
	}
	logrus.Info("Rlimit final: ", rLim)

	// logrus.SetLevel(logrus.DebugLevel)
	logrus.SetLevel(logrus.InfoLevel)
	logrus.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: "2006-01-02T15:04:05.000000000",
	})
	os.MkdirAll("./logs", 0777)
	filename := fmt.Sprintf("./logs/tide-%s.log", time.Now().Format("20060102-150405"))
	logFile, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		logrus.Fatal("open log file failed, ", err)
	}

	logrus.SetOutput(logFile)
}

func initFlags() *tideServeConfig {
	flag.Parse()
	cpuNum = runtime.NumCPU()
	runtime.GOMAXPROCS(cpuNum)

	selfRole := clsstat.ParseRole(roleParam)
	if selfRole == clsstat.UNKNOWN {
		logrus.Fatalf("unknown role type, `%s`", roleParam)
	}

	reqNum := cpuNum / conc
	dev.SetCPUReqNum(reqNum)

	// build self infomation
	selfId := stringx.GenerateId()
	if nodeName != "" {
		selfId = nodeName
	}
	selfIp := host
	selfAddr := stringx.Concat(selfIp, ":", strconv.FormatInt(int64(port), 10))
	// TODO: configure GPU Num
	selfRes := &routine.Resource{CpuNum: cpuNum, GpuNum: 0}

	logrus.Info("GOMAXPROCS: ", runtime.GOMAXPROCS(0))
	logrus.Info("SelfID:", selfId)
	logrus.Info("SelfAddr:", selfAddr)
	logrus.Info("SelfCpuNum:", cpuNum)
	logrus.Info("SelfRole:", selfRole)
	logrus.Info("PerStubReq:", dev.GetStubResource().CpuNum)
	logrus.Info("PerReqReq:", dev.GetReqResource().CpuNum)

	clsstat.Init(selfId, selfAddr, selfRole, selfRes)
	containerx.InitCgroup()
	if selfRole == clsstat.EDGE {
		containerx.InitCombCgroup()
	}

	return &tideServeConfig{
		Host:      selfAddr,
		Port:      port,
		CpuNum:    cpuNum,
		Addresses: addresses,
		NodeName:  selfId,
		SelfRole:  selfRole,
		SelfRes:   selfRes,
	}

}

func serveTide() {
	initEnv()
	config := initFlags()

	procM := procmgr.NewProcMgr(config.SelfRes)
	stubM := stubmgr.NewStubMgr(procM)

	// setup event server
	evtDispr := event.NewEventDispatcher()
	if config.SelfRole != clsstat.CLOUD {
		stubmgrhdlr.RegisterEventHandler(evtDispr, stubM)
		clsstathdlr.RegisterEventHandler(evtDispr)
	} else {
		cloudhdlr.RegisterEventHandler(evtDispr, stubM)
	}
	evtSvr := server.NewEventServer(evtDispr)

	cmnct.Init(config.NodeName, config.Host, evtDispr)

	// setup grpc server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", config.Port))
	if err != nil {
		logrus.Fatalf("failed to listen: %v", err)
	}
	grpcSvr := grpc.NewServer()
	eventpb.RegisterEventServiceServer(grpcSvr, evtSvr)

	// start daemon goroutine
	go func() {
		logrus.Infof("grpc server listening at %v", lis.Addr())
		if err := grpcSvr.Serve(lis); err != nil {
			logrus.Fatalf("failed to serve: %v", err)
		}
	}()
	// only EDGE node boradcast its resource state
	if config.SelfRole == clsstat.EDGE {
		procM.StartHeartbeat()
	}
	broadcastJoin(config.NodeName, config.Host, config.SelfRole, config.SelfRes)

	// graceful exit
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGTERM, os.Interrupt)
	onErr := func(err error) {
		if err != nil {
			logrus.Error(err)
		}
	}

	exitCh := make(chan struct{})

	go func() {
		for {
			select {
			case sig := <-c:
				logrus.Infof("Got %s signal. Aborting...", sig)
				close(exitCh)
				return
			case <-time.After(10 * time.Second):
				// default:
				// 	time.Sleep(10 * time.Second)
				// 	// logrus.Info("RUNNING and wait for exit signal")
			}
		}
	}()

	<-exitCh
	grpcSvr.GracefulStop()
	onErr(stubM.Close())
	onErr(procM.Close())
	containerx.DeleteAllCgroup()
	if config.SelfRole == clsstat.EDGE {
		containerx.DeleteAllCombCgroup()
	}
}

func broadcastJoin(selfId, selfAddr string, role clsstat.NodeRole, selfRes *routine.Resource) {
	servers := strings.Split(addresses, ",")

	ndNewEvtPb := &eventpb.NodeNewEvent{
		NodeId:        selfId,
		Address:       selfAddr,
		Role:          int32(role),
		TotalResource: &commpb.Resource{CpuNum: int32(selfRes.CpuNum), GpuNum: int32(selfRes.GpuNum)},
		IsReply:       false,
	}
	content, err := proto.Marshal(ndNewEvtPb)
	if err != nil {
		logrus.Fatal(err)
	}
	evt := &event.Event{Type: interfaces.NodeNewEvt, Content: content}
	for _, server := range servers {
		if server == "" {
			continue
		}
		if err := cmnct.Singleton().SendToAddress(server, evt); err != nil {
			logrus.Fatal(err)
		}
	}
}

// Get preferred outbound ip of this machine
func getHost() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)

	return localAddr.IP.String()
}
