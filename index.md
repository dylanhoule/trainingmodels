---
title: Sketch vs. Network
tag: ml
blurb: Draw a doodle and a simple CNN trained in PyTorch guesses it live right in your browser.
hue: 43, 82, 224
order: 1
image: screenshot.png
stack: PyTorch, ONNX, onnxruntime-web, vanilla JS
live: https://dylanhoule.github.io/sketch-guesser/
repo: https://github.com/dylanhoule/sketch-guesser
---

A Pictionary-playing neural network you can actually try. I trained a small
convolutional net in PyTorch on 120,000 QuickDraw doodles, exported it to ONNX,
and it runs entirely in your browser via WebAssembly — it guesses *while you're
still drawing*, and your sketch never leaves the page.

The hard part wasn't the model (93.4% test accuracy after three epochs) — it was
the gap between clean training data and a stranger's mouse-drawn scribble: the
canvas preprocessing has to reproduce the dataset's centering, scaling, and
stroke weight exactly, or the model sees garbage no matter how well it trained.

## Highlights
- 93.4% test accuracy from a 422k-parameter CNN that ships as a 1.7 MB file — small enough to load like an image
- Live inference on every stroke via `onnxruntime-web`, throttled so predictions never queue behind a fast pen
- Canvas preprocessing that mirrors training exactly: the stroke alpha channel *is* the white-on-black input, antialiasing included
- Training data fetched with HTTP Range requests — ~9 MB per category instead of the full ~100 MB files
- Zero backend: PyTorch → ONNX → WASM, hosted as three static files on GitHub Pages
