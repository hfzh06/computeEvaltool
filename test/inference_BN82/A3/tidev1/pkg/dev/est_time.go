package dev

import "time"

func GetEstExecTime(nid string) int {
	return int(250 * time.Millisecond)
}

func GetToleranceTime() int {
	return int(300 * time.Millisecond)
}
