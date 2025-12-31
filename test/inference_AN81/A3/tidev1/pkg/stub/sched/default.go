package sched

import (
	"math/rand"
	"sort"
	"time"

	"github.com/ict/tide/pkg/clsstat"
	"github.com/ict/tide/pkg/dev"
	"github.com/ict/tide/pkg/interfaces"
	"github.com/ict/tide/pkg/routine"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func defaultSortStub(rtn *routine.Routine, stStats []*clsstat.StubState) []*clsstat.StubState {
	sort.Slice(stStats, func(i, j int) bool {
		return stStats[i].CreateTime.Before(stStats[j].CreateTime)
	})

	estExecTime := dev.GetEstExecTime(clsstat.Singleton().GetSelf().Id)
	endBefore := rtn.SubmitTime + int64(dev.GetToleranceTime())
	// TODO: 直接返回空
	if endBefore < time.Now().UnixNano() {
		return make([]*clsstat.StubState, 0)
	}

	var filtered []*clsstat.StubState
	for _, stStat := range stStats {
		if stStat.Mode == interfaces.StubEntry || int(stStat.EstEndTime())+estExecTime > int(endBefore) {
			continue
		}
		filtered = append(filtered, stStat)
	}
	return filtered
}

func defaultSortNode(rtn *routine.Routine, ndStats []*clsstat.NodeState) []*clsstat.NodeState {
	return cpuBalanceSortNode(rtn, ndStats)
	// return randomSortNode(rtn, ndStats)
	// var filtered []*clsstat.NodeState
	// for i := len(ndStats) - 1; i >= 0; i-- {
	// 	ndStat := ndStats[i]
	// 	if !ndStat.ResRemain.Greater(routine.EmptyResourse) {
	// 		continue
	// 	}
	// 	filtered = append(filtered, ndStat)
	// }
	// return filtered
}

func cpuBalanceSortNode(rtn *routine.Routine, ndStats []*clsstat.NodeState) []*clsstat.NodeState {
	type pair struct {
		util float32
		node *clsstat.NodeState
	}

	var pairs []*pair
	for i := len(ndStats) - 1; i >= 0; i-- {
		ndStat := ndStats[i]
		if !ndStat.ResRemain.Greater(routine.EmptyResourse) {
			continue
		}
		util := float32(ndStat.ResRemain.CpuNum) / float32(ndStat.ResTotal.CpuNum)
		pairs = append(pairs, &pair{util, ndStat})
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].util > pairs[j].util
	})

	ret := make([]*clsstat.NodeState, len(pairs))
	for i, p := range pairs {
		ret[i] = p.node
	}
	return ret
}

func randomSortNode(rtn *routine.Routine, ndStats []*clsstat.NodeState) []*clsstat.NodeState {
	var filtered []*clsstat.NodeState
	for i := len(ndStats) - 1; i >= 0; i-- {
		ndStat := ndStats[i]
		if !ndStat.ResRemain.Greater(routine.EmptyResourse) {
			continue
		}
		filtered = append(filtered, ndStat)
	}

	rand.Shuffle(len(filtered), func(i, j int) { filtered[i], filtered[j] = filtered[j], filtered[i] })
	return filtered
}
