#!/bin/bash
echo "${@:1:$#-1}"
echo "${@: -1}"