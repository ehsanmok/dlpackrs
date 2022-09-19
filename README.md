# dlpackrs

[![Build](https://github.com/ehsanmok/dlpackrs/actions/workflows/build.yml/badge.svg)](https://github.com/ehsanmok/dlpackrs/actions/workflows/build.yml)
[![miri-checked](https://img.shields.io/badge/miri-checked-green)](https://img.shields.io/badge/miri-checked-green)
[![crates.io](https://img.shields.io/crates/v/dlpackrs.svg)](https://crates.io/crates/dlpackrs)
[![docs.rs](https://docs.rs/dlpackrs/badge.svg)](https://docs.rs/dlpackrs)

This crate provides a safe idiomatic Rust binding to [DLPack](https://dmlc.github.io/dlpack/latest/) which is the standard in-memory, (mostly) hardware agnostic data format , recognized by major Deep Learning frameworks such as [PyTorch](https://pytorch.org/docs/stable/dlpack.html), [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/from_dlpack), [MXNet](https://mxnet.apache.org/versions/master/api/python/docs/_modules/mxnet/dlpack.html), [TVM](https://tvm.apache.org/docs/reference/api/python/contrib.html#module-tvm.contrib.dlpack) and major array processing frameworks such as [NumPy](https://numpy.org/doc/stable/release/1.22.0-notes.html#add-nep-47-compatible-dlpack-support) and [CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.fromDlpack.html). An important feature of this standard is to provide *zero-cost* tensor conversion across frameworks on a particular supported hardware.

The Minimum Supported Rust Version (MSRV) is the stable toolchain **1.57.0**.

## Usage

There are two main cases related to where the owner of the underlying data / storage of a tensor resides and what kind of operations are to be done.

### Memory Managed Tensor

In this case, `ManagedTensor` is built from `ManagedTensorProxy` which is a safe proxy for the unsafe `ffi::DLManagedTensor`.

### Non-Memory Manged Tensor (View)

In this case, the (invariant) Rust wrapper `Tensor` can be used or if needed the unsafe `ffi::DLTensor`.

## Example

When ownership is concerned, one can use the `ManagedTensor`. Here is an example on how the bi-directional conversion

<div align="center">ndarray::ArrayD <--> ManagedTensor</div>

is done at zero-cost.

```rust
impl<'tensor, 'ctx> From<&'tensor mut ArrayD<f32>> for ManagedTensor<'tensor, 'ctx> {
    fn from(t: &'tensor mut ArrayD<f32>) -> Self {
        let dlt: Tensor<'tensor> = Tensor::from(t);
        let inner = DLManagedTensor::new(dlt.0, ptr::null_mut());
        ManagedTensor(inner)
    }
}

impl<'tensor, 'ctx> From<&mut ManagedTensor<'tensor, 'ctx>> for ArrayD<f32> {
    fn from(mt: &mut ManagedTensor<'tensor, 'ctx>) -> Self {
        let dlt: DLTensor = mt.0.inner.dl_tensor.into();
        unsafe {
            let arr = RawArrayViewMut::from_shape_ptr(dlt.shape().unwrap(), dlt.data() as *mut f32);
            arr.deref_into_view_mut().into_dyn().to_owned()
        }
    }
}
```

And when ownership is not concerned, one can use `Tensor` as a view. Here is an example on how the bi-directional converion

<div align="center">ndarray::ArrayD <--> Tensor</div>

is done at zero-cost.

```rust
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
```

See the complete [examples/sample](https://github.com/ehsanmok/dlpackrs/blob/main/examples/sample/src/main.rs) where the above cases have been simulated for the Rust [ndarray](https://docs.rs/ndarray/latest/ndarray/) conversion.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
