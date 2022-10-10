use core::ptr::NonNull;
use std::marker::PhantomData;


pub struct GridLinkedList<T> {
    anchor: Node<T>,
    _boo: PhantomData<T>,
}

type Link<T> = Option<NonNull<Node<T>>>;

struct Node<T> {
    top: Link<T>,
    bottom: Link<T>,
    left: Link<T>,
    right: Link<T>,
    element: T,
}

enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

impl<T> GridLinkedList<T> {
    pub fn new(start: T) -> Self {
        Self {
            anchor: Node::make_anchor(start),
            _boo: PhantomData,
        }
    }

    pub fn add_at(&mut self, x: i32, y: i32){
        // problem with stitching new chunks with existing (if path to the existing is not trivial)
        // ****_
        // *__+_  the + has to have references to both: top and bottom
        // ****_
    }
}

impl<T> Node<T> {
    fn make_anchor(element: T) -> Self {
        Self {
            top: None,
            bottom: None,
            left: None,
            right: None,
            element,
        }
    }
    fn new(element: T, parent: &Node<T>, side: Side) -> Self {
        let mut top: Link<T> = None;
        let mut bottom = None;
        let mut left = None;
        let mut right = None;

        match side {
            Side::Top => top = NonNull::new(parent as *mut _),
            Side::Bottom => bottom = NonNull::new(parent as *mut _),
            Side::Left => left = NonNull::new(parent as *mut _),
            Side::Right => right = NonNull::new(parent as *mut _),
        }
        Self {
            top,
            bottom,
            left,
            right,
            element,
        }
    }

    fn append(&mut self, element: T, side: Side) {
        let child = Self::new(element, self, side);
        match side {
            Side::Top => self.top = NonNull::new(*child),
            Side::Bottom => self.bottom = NonNull::new(*child),
            Side::Left => self.left = NonNull::new(*child),
            Side::Right => self.right = NonNull::new(*child),
        }
    }
}