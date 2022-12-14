#![allow(clippy::drop_ref)]

use std::ffi::c_void;

use dlpackrs::{DataType, Device, ManagedTensor as DLManagedTensor, Tensor as DLTensor};
use ndarray::{Array, ArrayD, RawArrayViewMut};

#[derive(Debug)]
pub struct Tensor<'tensor>(DLTensor<'tensor>);

impl<'tensor> From<&'tensor mut ArrayD<f32>> for Tensor<'tensor> {
    fn from(arr: &'tensor mut ArrayD<f32>) -> Self {
        let inner = DLTensor::new(
            arr.as_mut_ptr() as *mut c_void,
            Device::default(),
            arr.ndim() as i32,
            DataType::f32(),
            arr.shape().as_ptr() as *const _ as *mut i64,
            arr.strides().as_ptr() as *const _ as *mut i64,
            0,
        );
        Tensor(inner)
    }
}

impl<'tensor> From<&'tensor mut Tensor<'tensor>> for ArrayD<f32> {
    fn from(t: &'tensor mut Tensor<'tensor>) -> Self {
        unsafe {
            let arr = RawArrayViewMut::from_shape_ptr(t.0.shape().unwrap(), t.0.data() as *mut f32);
            arr.deref_into_view_mut().into_dyn().to_owned()
        }
    }
}

// The context holds DLManagedTensor
#[derive(Debug)]
pub struct ManagedContext<'tensor, C>(DLManagedTensor<'tensor, C>);

impl<'tensor, C> From<&'tensor mut ArrayD<f32>> for ManagedContext<'tensor, C> {
    fn from(t: &'tensor mut ArrayD<f32>) -> Self {
        let dlt: Tensor<'tensor> = Tensor::from(t);
        let inner = DLManagedTensor::new(dlt.0, None);
        ManagedContext(inner)
    }
}

impl<'tensor, C> From<&mut ManagedContext<'tensor, C>> for ArrayD<f32> {
    fn from(mt: &mut ManagedContext<'tensor, C>) -> Self {
        let dlt: DLTensor = mt.0.inner.dl_tensor.into();
        unsafe {
            let arr = RawArrayViewMut::from_shape_ptr(dlt.shape().unwrap(), dlt.data() as *mut f32);
            arr.deref_into_view_mut().into_dyn().to_owned()
        }
    }
}

fn main() {
    let mut ping = Array::from_shape_vec((2, 3), vec![1f32, 2., 3., 4., 5., 6.])
        .unwrap()
        .into_dyn();
    println!("ping {:?}", ping);
    let mut tensor: Tensor<'_> = Tensor::from(&mut ping);
    println!(
        "tensor {:?} with shape {:?}, itemsize {:?} bytes, strides {:? }, total memory size {:?} bytes",
        tensor,
        tensor.0.shape().unwrap(),
        tensor.0.itemsize(),
        tensor.0.strides().unwrap(),
        tensor.0.size().unwrap(),
    );
    let pong = ArrayD::from(&mut tensor);
    println!("pong {:?}", pong);
    assert!(pong.into_dyn().abs_diff_eq(&ping, 1e-8f32));
    let mut managed_tensor: ManagedContext<f32> = (&mut ping).into();
    println!("managed tensor {:?}", managed_tensor);
    let deleter: fn(&mut DLManagedTensor<f32>) = |managed_tensor| {
        println!("manager tensor deleter is called");
        drop(managed_tensor);
    };
    managed_tensor.0.set_deleter(deleter);
    println!("managed tensor with deleter {:?}", managed_tensor);
    let managed_pong: ArrayD<f32> = ArrayD::from(&mut managed_tensor);
    assert!(managed_pong.abs_diff_eq(&ping, 1e-8f32));
}
