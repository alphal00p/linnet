use duplicate::duplicate;
use std::borrow::Borrow;
pub trait RefZero<T = Self>: Borrow<T> {
    fn ref_zero(&self) -> T;
}

pub trait RefOne {
    fn ref_one(&self) -> Self;
}

#[cfg(feature = "symbolica")]
impl RefZero for Atom {
    fn ref_zero(&self) -> Self {
        Atom::new_num(0)
    }
}

#[cfg(feature = "symbolica")]
impl<T: RefOne + Real + RefZero> RefOne for symbolica::domains::float::Complex<T> {
    fn ref_one(&self) -> Self {
        Self::new(self.re.ref_one(), self.im.ref_zero())
    }
}

#[cfg(feature = "symbolica")]
impl<T: Real + RefZero> RefZero for symbolica::domains::float::Complex<T> {
    fn ref_zero(&self) -> Self {
        Self::new(self.re.ref_zero(), self.im.ref_zero())
    }
}

// impl<T: num::Zero> RefZero for T { future impls Grrr
//     fn zero(&self) -> Self {
//         T::zero()
//     }
// }

duplicate! {
    [types zero one;
        [f32] [0.0] [1.];
        [f64] [0.0] [1.];
        [i8] [0] [1];
        [i16] [0] [1] ;
        [i32] [0] [1];
        [i64] [0] [1];
        [i128] [0] [1];
        [u8] [0] [1];
        [u16] [0] [1];
        [u32] [0] [1];
        [u64] [0] [1];
        [u128] [0] [1];
        ]

    impl RefZero for types{
        fn ref_zero(&self)-> Self{
            zero
        }
    }

    impl RefZero<types> for &types{
        fn ref_zero(&self)-> types{
            zero
        }
    }

    impl RefOne for types{
        fn ref_one(&self)-> Self{
            one
        }
    }
}
