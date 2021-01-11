# Linear Algebra Notes

# Week 1: Syllabus Week and Vector Stuff!

## Notation

* Vector: Ordered list of numbers
* n-vector: vector with $n$ entries
* $i$th element: $a_i$
	* indexes run from 1 to $n$
* $a = b$ if they are the same length and $a_i = b_i$ for all $i$


## Block Vectors

A block vector (or stacked vector) is a concatenation of vectors 

## Special Vectors

* n-vectors
	* All Entries 0: Denoted as $0_n$ or just 0
	* All entries 1: Denoted as **$1_n$** or just **1**
	* A unit vector has all entries 0 except for one
* A unit vector has all entries 0 except for one
	* denoted $e_i$ where the $i$th intry is 1


## Sparsity

* Vector with many entries 0
* Stored very effeciently on a computer
* $\mathrm{nnz}(x)$ is the number of nonzero


## Vectors as representation

* Location/displacement
* Physics
* Colors
* Images
* Word counts

Vectors can represent all sorts of things, which is why LA is so important!


## Vector Addition

$n$-vectors $a$ and $b$ are added via elementwise addition, denoted $a + b$

* commutative 
* associative
* identity
* zero

All just the same as normal addition. Vector addition can be viewed as the total displacement of the system, viewing from a visual standpoint

## Scalar multiplication

Elementwise multiplication by a scalar

* Denoted $a\beta$ or $\beta a$
* Associative and left/right distributive
