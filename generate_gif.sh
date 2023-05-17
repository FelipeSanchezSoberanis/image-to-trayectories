#!/usr/bin/env bash

convert -delay 16 -loop 0 animation/frame-*.png animation/output.gif
rm animation/*.png

