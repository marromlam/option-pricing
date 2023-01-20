#!/bin/bash -eu

SYMBOL=$1
INTERVAL=$2
STARTDATE=$3
ENDDATE=$4
FILENAME=$SYMBOL.csv

STARTEPOCH=$(gdate -d "$STARTDATE" '+%s')
ENDEPOCH=$(gdate -d "$ENDDATE" '+%s')

cookieJar=$(mktemp)
function cleanup() {
	rm $cookieJar
}
trap cleanup EXIT

function parseCrumb() {
	sed 's+}+\n+g' | grep CrumbStore | cut -d ":" -f 3 | sed 's+"++g'
}

function extractCrumb() {
	crumbUrl="https://ca.finance.yahoo.com/quote/$SYMBOL/history?p=$SYMBOL"
	curl -s --cookie-jar $cookieJar $crumbUrl | parseCrumb
}

crumb=$(extractCrumb)

baseUrl="https://query1.finance.yahoo.com/v7/finance/download/"
args="$SYMBOL?period1=$STARTEPOCH&period2=$ENDEPOCH&interval=$INTERVAL"
crumbArg="&crumb=$crumb"
sheetUrl="$baseUrl$args&events=history$crumbArg"

curl -s --cookie $cookieJar "$sheetUrl" >$FILENAME
