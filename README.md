# Cuda bitonic sorting
This project was developed within the scope of a university project.

The goal is to implement bitonic sort in cuda C, to sort an array of integers with the GPU.
There are two "programs":
- prog1: sorts in a row manner, meaning it goes through the array linearly
- prog2: sorts in a column manner, meaning it jumps from element to element skiping the an entire line to go to the next column. For instance, if the data is seen as a 10x10 matrix, then the algorithm compares the element 0 with the element 9, then the element 1 with the element 11, and so on...

The point is to confirm the performance between both and how the cache affects it.
For more information, read [the presentation](https://github.com/pompeucosta/CLE-Assignment3/blob/main/present.pdf).
