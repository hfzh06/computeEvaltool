package dev

import "github.com/ict/tide/pkg/routine"

var (
	cpuReqNum = 8
	gpuReqNum = 0
)

func SetCPUReqNum(num int) {
	cpuReqNum = num
}

func SetGPUReqNum(num int) {
	gpuReqNum = num
}

func GetReqResource() *routine.Resource {
	return &routine.Resource{CpuNum: cpuReqNum, GpuNum: gpuReqNum}
	// return &routine.Resource{CpuNum: 4, GpuNum: 0}
}

func GetStubResource() *routine.Resource {
	return &routine.Resource{CpuNum: cpuReqNum, GpuNum: gpuReqNum}
	// return &routine.Resource{CpuNum: 4, GpuNum: 0}

}
