#![feature(associated_type_bounds)]

struct SliceTuple<'a, T0, T1> {
    arrays : (&'a [T0], &'a [T1]), // (&[T]...)
    base : usize,
    len : usize,
}

// Implements Iterator for types with a Slice interface
impl<'a,T0:Copy,T1:Copy> Iterator for SliceTuple<'a, T0, T1> {
    type Item = <Self as Value<usize>>::Output;
    // Returns next item by value (SliceTuple index returns temporary reference wrapper)
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 { None? }; // todo: fast path
        let next = self.value(0);
        *self = self.value(1..);
        Some(next.into())
    }
}

/// std::ops::Index by value to allow returning value types (could not return by reference eventhough [] dereferences *index immediately)
trait Value<T> {
    type Output;
    fn value(&self, idx: T) -> Self::Output;
}
/// Presents tuple of slices as a slice of tuples
impl<'a, T0 : Copy, T1 : Copy> Value<usize> for SliceTuple<'a, T0, T1> {
    type Output = (T0, T1); //(T)...;
    fn value(&self, idx: usize) -> Self::Output {
        let index = self.base+idx;
        (self.arrays.0[index], self.arrays.1[index])
    }
}
/// Slices tuple of slices
impl<'a, T0, T1> Value<std::ops::RangeFrom<usize>> for SliceTuple<'a, T0, T1> {
    type Output = SliceTuple<'a, T0, T1>;
    fn value(&self, idx: std::ops::RangeFrom<usize>) -> Self::Output { SliceTuple{ arrays: self.arrays, base: self.base+idx.start, len: self.len-idx.start } }
}

/// [] trait similar to core impl<T> [T]. Except Index trait cannot be implemented by SliceTuple, replaced by Value trait
trait Slice<T> : Value<usize,Output=T> + Value<std::ops::RangeFrom<usize>> + From<<Self as Value<std::ops::RangeFrom<usize>>>::Output>/* *self=self.value(..)*/ {
    fn len(&self) -> usize;
}
/// Length of wrapped slice view. Extra {base, len} should be easier to strength reduce than unused field slicing
impl<'a, T0 : Copy, T1 : Copy> Slice<(T0,T1)> for SliceTuple<'a, T0, T1> {
    fn len(&self) -> usize { self.len }
}

trait Split : Iterator  {
    /// Consumes the sequence until first item matching predicate.
    /// Assumes predicate splits the sequence in [false[..], true[..]] to use binary search.
    /// Assumes splits are more likely to be close to the start: try small splits first.
    fn split<P : Fn(Self::Item) -> bool>(&mut self, predicate : P) -> Option<Self::Item>;
}
// Implements Split for sequences with a Slice interface
impl<T : Iterator+Slice<Self::Item>> Split for T { //where Slice::Item: Iterator::Item {
  fn split<P : Fn(Self::Item) -> bool>(&mut self, predicate : P) -> Option<Self::Item> {
        // first bottom-up left skewed split. Skips redundant top levels, more efficient for j < 2^(log2(B)/2))
        let mut size = 1;
        loop {
            if size >= self.len() { size = self.len(); break; } // todo: ensure B.len > initial split from parent
            if predicate(self.value(size)) { size=size+1; break; } // left
            *self = self.value(size+1..).into(); /*self.nth(size)*/ // right
            size *= 2;
            if size >= self.len() { size = self.len(); break; } // rightmost
        }
        // then standard top-down half splits
        while size > 1 {
            let split = size / 2;
            if predicate(self.value(split)) { assert_eq!(size-split, split); size=split; }
            else { size -= split; *self = self.value(split..).into(); /*self.nth(split-1)*/ }
        }
        if !predicate(self.value(0)) { self.next(); } //*self = self.value(1..).into(); }
        self.next() // todo? unchecked fast path
    }
}

trait Component : Copy {
    type Index : Ord+Default;
    fn index(self) -> Self::Index;
    type Value;
    fn value(self) -> Self::Value;
}
impl<I : Ord+Default+Copy, V: Copy> Component for (I, V) {
    type Index = I;
    fn index(self) -> Self::Index { self.0 }
    type Value = V;
    fn value(self) -> Self::Value{ self.1 }
}

// Finds next index
trait FindNext : Split<Item:Component> {
    fn find_next(&mut self, index : <Self::Item as Component>::Index) -> Option<Self::Item> {
        self.split(|x:Self::Item| { /*println!("{:?} <= {:?}", index, x.index());*/ index <= x.index() })//.into()
    }
}
impl<T:Split<Item:Component>> FindNext for T {}

/// Generic sparse vector joint non zero iteration
/// Skips runs of non-zero values absent from the other vector with skewed binary search
/// Worst case: min(A,B) log max(A,B)
#[allow(non_snake_case)]
fn for_each<S : FindNext<Item:Component>, F : FnMut(<S::Item as Component>::Value, <S::Item as Component>::Value)>(mut A : S, mut B : S, mut f : F) -> Option<()> {
    let mut a = A.find_next( <S::Item as Component>::Index::default() )?;
    loop {
        let mut b = B.find_next( a.index() )?; // next B component
        while a.index() == b.index() { // Lockstep fast path
            f(a.value(), b.value()); // Folds matching components
            a = A.next()?;
            b = B.next()?;
        }
        // A <=> B (swap vector roles to skip any long runs of non-zero values in A absent from B)
        a = A.find_next( b.index() )?; // next A component
        while a.index() == b.index() { // Lockstep fast path
            f(a.value(), b.value()); // Folds matching components
            a = A.next()?;
            b = B.next()?;
        }
    }
}

fn fold<S : FindNext<Item:Component>, Acc, F : Fn(&Acc, <S::Item as Component>::Value, <S::Item as Component>::Value) -> Acc>(a : S, b: S, mut acc : Acc, f : F) -> Acc {
    for_each(a, b, |a, b| { acc=f(&acc, a, b); });
    acc
}

fn dot<S : FindNext<Item:Component>>(a : S, b : S) -> f32 where <S::Item as Component>::Value : Into<f32> {
    fold(a, b, 0., |sum, a, b| { sum + a.into() * b.into() })
}

struct SparseVec<I,V>(Vec<I>, Vec<V>);

impl<'a, I, V> Value<std::ops::RangeFull> for &'a SparseVec<I, V> {
    type Output = SliceTuple<'a, I, V>;
    fn value(&self, _: std::ops::RangeFull) -> Self::Output { SliceTuple{ arrays: (&self.0, &self.1), base: 0, len: self.0.len() } }
}

fn sparse<V : Default+PartialEq+Copy>(dense : &[V]) -> SparseVec<u32, V> {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (i, v) in dense.iter().enumerate() { if *v != V::default() { indices.push(i as u32); values.push(*v); } }
    SparseVec(indices, values)
}

fn main() {
    let v0 = &sparse(&[1., 4., 0., 0., 8., 9.]);
    let v1 = &sparse(&[0., 0., 5., 0., 1., 0.]);
    assert_eq!(dot(v0.value(..), v1.value(..)), 8.);
}
