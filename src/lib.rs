//! [![miri-checked](https://img.shields.io/badge/miri-checked-green)](https://img.shields.io/badge/miri-checked-green)
//! [![crates.io](https://img.shields.io/crates/v/dlpackrs.svg)](https://crates.io/crates/dlpackrs)
//! [![docs.rs](https://docs.rs/dlpackrs/badge.svg)](https://docs.rs/dlpackrs)
//! [![GitHub](https://img.shields.io/crates/l/dlpackrs)](https://github.com/ehsanmok/dlpackrs)
//!
//! <br>
//!
//! This crate provides a safe idiomatic Rust binding to [DLPack](https://dmlc.github.io/dlpack/latest/)
//! which is the standard in-memory, (mostly) hardware agnosnic data exchange interface
//! and is recognized by major Deep Learning frameworks such as [PyTorch](https://pytorch.org/docs/stable/dlpack.html),
//! [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/from_dlpack),
//! [MXNet](https://mxnet.apache.org/versions/master/api/python/docs/_modules/mxnet/dlpack.html),
//! [TVM](https://tvm.apache.org/docs/reference/api/python/contrib.html#module-tvm.contrib.dlpack)
//! and major array processing frameworks such as [NumPy](https://numpy.org/doc/stable/release/1.22.0-notes.html#add-nep-47-compatible-dlpack-support)
//! and [CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.fromDlpack.html).
//! An important feature of this standard is to provide *zero-cost* tensor conversion across frameworks on a particular supported hardware.
//!
//! The Minimum Supported Rust Version (MSRV) is the stable toolchain **1.57.0**.
//!
//! ## Usage
//!
//! There are two usages related to where the owner of the underlying data / storage of a tensor resides.
//! The main distinction is when the underlying data gets dropped and which framework is responsible for it.
//!
//! ### Rust as the owner
//!
//!In this case, `ManagedTensor` is built from `ManagedTensorProxy` which is the same proxy for `ffi::DLManagedTensor`.
//!
//! ### Non-Rust owner
//!
//! In this case, either `ffi::DLTensor` or its (invariant) Rust wrapper `Tensor` can be used.
//!
//! <br>
//!
//! ## Example
//!
//! For when ownership is not concerned, one can use `Tensor`. Here is an example on how the conversion
//!
//! <div align="center">ndarray::ArrayD <--> Tensor</div>
//!
//! can happen at zero-cost.
//!
//! ```no_run
//! impl<'tensor> From<&'tensor mut ArrayD<f32>> for Tensor<'tensor> {
//!     fn from(arr: &'tensor mut ArrayD<f32>) -> Self {
//!         let inner = DLTensor::new(
//!             arr.as_mut_ptr() as *mut c_void,
//!             Device::default(),
//!             arr.ndim() as i32,
//!             DataType::f32(),
//!             arr.shape().as_ptr() as *const _ as *mut i64,
//!             arr.strides().as_ptr() as *const _ as *mut i64,
//!             0,
//!         );
//!         Tensor(inner)
//!     }
//! }
//!
//! impl<'tensor> From<&'tensor mut Tensor<'tensor>> for ArrayD<f32> {
//!    fn from(t: &'tensor mut Tensor<'tensor>) -> Self {
//!         unsafe {
//!             let arr = RawArrayViewMut::from_shape_ptr(t.0.shape().unwrap(), t.0.data() as *mut f32);
//!             arr.deref_into_view_mut().into_dyn().to_owned()
//!         }
//!     }
//! }
//! ```
//!
//! And when ownership is concerned, one can use the `ManagedTensor`. Here is an example on how the conversion
//!
//! <div align="center">ndarray::ArrayD <--> ManagedTensor</div>
//!
//! can happen at zero-cost.
//!
//! ```no_run
//! impl<'tensor, 'ctx> From<&'tensor mut ArrayD<f32>> for ManagedTensor<'tensor, 'ctx> {
//!     fn from(t: &'tensor mut ArrayD<f32>) -> Self {
//!         let dlt: Tensor<'tensor> = Tensor::from(t);
//!         let inner = DLManagedTensor::new(dlt.0, ptr::null_mut());
//!         ManagedTensor(inner)
//!     }
//! }
//!
//! impl<'tensor, 'ctx> From<&mut ManagedTensor<'tensor, 'ctx>> for ArrayD<f32> {
//!     fn from(mt: &mut ManagedTensor<'tensor, 'ctx>) -> Self {
//!         let dlt: DLTensor = mt.0.inner.dl_tensor.into();
//!         unsafe {
//!             let arr = RawArrayViewMut::from_shape_ptr(dlt.shape().unwrap(), dlt.data() as *mut f32);
//!             arr.deref_into_view_mut().into_dyn().to_owned()
//!         }
//!     }
//! }
//! ```
//!
//! See the complete [examples/sample](https://github.com/ehsanmok/dlpackrs/blob/main/examples/sample/src/main.rs)
//! where the above cases have been simulated for the Rust [ndarray](https://docs.rs/ndarray/latest/ndarray/) conversion.

#![allow(clippy::missing_safety_doc)]
pub mod ffi {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]
    pub use dlpack_sys::*;
}

pub mod datatype;
pub mod device;
pub mod errors;
pub mod tensor;

pub use datatype::{DataType, DataTypeCode};
pub use device::{Device, DeviceType};
pub use tensor::{ManagedTensor, ManagedTensorProxy, Tensor};

pub fn version() -> u32 {
    ffi::DLPACK_VERSION
}

pub fn abi_version() -> u32 {
    ffi::DLPACK_ABI_VERSION
}
