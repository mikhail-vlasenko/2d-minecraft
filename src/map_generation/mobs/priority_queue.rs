use std::cmp::Ordering;
use std::collections::BinaryHeap;


#[derive(Copy, Clone, Eq, PartialEq)]
struct QueueRecord<T: Eq> {
    priority: i32,  // lower is served first
    value: T
}

impl<T: Eq> Ord for QueueRecord<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

impl<T: Eq> PartialOrd for QueueRecord<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Lower priority is served first
pub struct PriorityQueue<T: Eq> {
    heap: BinaryHeap<QueueRecord<T>>
}

impl<T: Eq> PriorityQueue<T> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new()
        }
    }
    pub fn push(&mut self, priority: i32, value: T) {
        self.heap.push(QueueRecord { priority, value })
    }

    pub fn pop(&mut self) -> Option<T> {
        if let Some(QueueRecord { priority: _, value}) = self.heap.pop() {
            Some(value)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}
