use pin_project::{pin_project, pinned_drop};

use core::slice;
use std::{marker::PhantomData, os::raw::c_void, pin::Pin};

use crate::{
    datatype::DataType,
    device::Device,
    ffi::{DLManagedTensor, DLTensor},
};

/// Non-owned Tensor type interface.
/// See [DLTensor](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv48DLTensor)
#[derive(Debug)]
#[repr(transparent)]
pub struct Tensor<'tensor> {
    pub inner: DLTensor,
    _marker: PhantomData<fn(&'tensor ()) -> &'tensor ()>, // invariant wrt 'tensor
}

impl<'tensor> From<Tensor<'tensor>> for DLTensor {
    fn from(ts: Tensor<'tensor>) -> Self {
        ts.inner
    }
}

impl<'tensor> From<DLTensor> for Tensor<'tensor> {
    fn from(dts: DLTensor) -> Self {
        Tensor {
            inner: dts,
            _marker: PhantomData,
        }
    }
}

impl<'tensor> Tensor<'tensor> {
    /// Constructor
    pub fn new(
        data: *mut c_void,
        device: Device,
        ndim: i32,
        dtype: DataType,
        shape: *mut i64,
        strides: *mut i64,
        byte_offset: u64,
    ) -> Self {
        let inner = DLTensor {
            data,
            device: device.into(),
            ndim,
            dtype: dtype.into(),
            shape,
            strides,
            byte_offset,
        };
        Tensor {
            inner,
            _marker: PhantomData,
        }
    }

    /// Returns the underlying DLTensor where lifetime parameter is removed.
    pub fn into_inner(self) -> DLTensor {
        self.inner
    }

    /// Consumes the Tensor and returns the raw pointer to its underlying DLTensor.
    pub fn into_raw(self) -> *const DLTensor {
        &self.inner as *const _
    }

    /// Creates a Tensor from a raw DLTensor pointer (must be non-null).
    pub unsafe fn from_raw(ptr: *mut DLTensor) -> Self {
        debug_assert!(!ptr.is_null());
        Tensor {
            inner: *ptr,
            _marker: PhantomData,
        }
    }

    /// Returns a *mut pointer to the underlying data of the Tensor.
    pub fn data(&self) -> *mut c_void {
        self.inner.data
    }

    /// Returns the device type.
    pub fn device(&self) -> Device {
        self.inner.device.into()
    }

    /// Returns the size of an entry/item in the Tensor.
    pub fn itemsize(&self) -> usize {
        let ty = self.dtype();
        ty.lanes() * ty.bits() / 8_usize
    }

    /// Returns the number of dimensions of the Tensor.
    pub fn ndim(&self) -> usize {
        self.inner.ndim as usize
    }

    /// Returns the type of the entries of the Tensor.
    pub fn dtype(&self) -> DataType {
        self.inner.dtype.into()
    }

    /// Returns the shape of the Tensor.
    pub fn shape(&self) -> Option<&[usize]> {
        let dlt = self.inner;
        if dlt.shape.is_null() || dlt.data.is_null() {
            return None;
        };
        let ret = unsafe { slice::from_raw_parts(dlt.shape as *const _, dlt.ndim as usize) };
        Some(ret)
    }

    /// Returns the strides of the underlying Tensor.
    pub fn strides(&self) -> Option<&[usize]> {
        let dlt = self.inner;
        if dlt.strides.is_null() || dlt.data.is_null() {
            return None;
        };
        let ret = unsafe { slice::from_raw_parts(dlt.strides as *const _, dlt.ndim as usize) };
        Some(ret)
    }

    /// Returns the byte offset of the underlying Tensor.
    pub fn byte_offset(&self) -> isize {
        self.inner.byte_offset as isize
    }

    /// Returns the size of the memory required to store the underlying data of the Tensor.
    pub fn size(&self) -> Option<usize> {
        let ty = self.dtype();
        self.shape().map(|v| {
            v.iter().product::<usize>() * (ty.bits() as usize * ty.lanes() as usize + 7) / 8
        })
    }
}

/// Safe proxy to ffi::DLManagedTensor which is self-referential by design.
/// See [DLManagedTensor](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv415DLManagedTensor)
#[derive(Debug)]
#[pin_project(PinnedDrop)]
pub struct ManagedTensorProxy<'ctx> {
    pub dl_tensor: DLTensor,
    #[pin]
    /// Holds the underlying DLTensor.
    pub manager_ctx: *mut c_void,
    #[pin]
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
    _marker: PhantomData<&'ctx ()>, // covariant wrt 'ctx
}

impl<'ctx> From<DLManagedTensor> for ManagedTensorProxy<'ctx> {
    fn from(dlmt: DLManagedTensor) -> Self {
        ManagedTensorProxy {
            dl_tensor: dlmt.dl_tensor,
            manager_ctx: dlmt.manager_ctx,
            deleter: dlmt.deleter,
            _marker: PhantomData,
        }
    }
}

impl<'ctx> From<ManagedTensorProxy<'ctx>> for DLManagedTensor {
    fn from(pmt: ManagedTensorProxy<'ctx>) -> Self {
        DLManagedTensor {
            dl_tensor: pmt.dl_tensor,
            manager_ctx: pmt.manager_ctx,
            deleter: pmt.deleter,
        }
    }
}

impl<'ctx> From<Pin<&mut ManagedTensorProxy<'ctx>>> for DLManagedTensor {
    fn from(pmt: Pin<&mut ManagedTensorProxy<'ctx>>) -> Self {
        DLManagedTensor {
            dl_tensor: pmt.dl_tensor,
            manager_ctx: pmt.manager_ctx,
            deleter: pmt.deleter,
        }
    }
}

impl<'ctx> ManagedTensorProxy<'ctx> {
    pub fn dl_tensor(&self) -> DLTensor {
        self.dl_tensor
    }

    pub fn manager_ctx(self: Pin<&mut Self>) -> *mut c_void {
        let this = self.project();
        *this.manager_ctx.get_mut()
    }

    pub fn set_manager_ctx(self: Pin<&mut Self>, manager_ctx: *mut c_void) {
        let mut this = self.project();
        *this.manager_ctx = manager_ctx;
    }

    pub fn deleter(
        self: Pin<&mut Self>,
    ) -> Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)> {
        let this = self.project();
        *this.deleter.get_mut()
    }
}

#[allow(clippy::needless_lifetimes)]
#[pinned_drop]
impl<'ctx> PinnedDrop for ManagedTensorProxy<'ctx> {
    fn drop(mut self: Pin<&mut Self>) {
        let mut dlm: DLManagedTensor = self.as_mut().into();
        if let Some(fptr) = self.deleter() {
            unsafe {
                let cfptr = std::mem::transmute::<*const (), unsafe fn(*mut DLManagedTensor)>(
                    fptr as *const (),
                );
                cfptr(&mut dlm as *mut _);
            };
        }
    }
}

/// ManagedTensor type with Rust as the main owner of the underlying data.
///
///  See [DLManagedTensor](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv415DLManagedTensor)
#[derive(Debug)]
#[repr(transparent)]
pub struct ManagedTensor<'tensor, 'ctx: 'tensor> {
    pub inner: ManagedTensorProxy<'ctx>,
    _marker: PhantomData<fn(&'tensor ()) -> &'tensor ()>,
}

impl<'tensor, 'ctx> ManagedTensor<'tensor, 'ctx> {
    pub fn new(tensor: Tensor<'tensor>, manager_ctx: *mut c_void) -> Self {
        let inner = ManagedTensorProxy {
            dl_tensor: tensor.into_inner(),
            manager_ctx,
            deleter: None,
            _marker: PhantomData,
        };

        ManagedTensor {
            inner,
            _marker: PhantomData,
        }
    }

    /// Sets a deleter function pointer
    pub fn set_deleter(&mut self, deleter: unsafe extern "C" fn(*mut DLManagedTensor)) {
        self.inner.deleter = Some(deleter);
    }

    /// Consumes the ManagedTensor and returns the raw pointer to its underlying DLManagedTensor.
    pub fn into_raw(self) -> *const DLManagedTensor {
        let ret: DLManagedTensor = self.inner.into();
        &ret as *const _
    }

    /// Returns a ManagedTensor instances from a raw pointer to DLManagedTensor.
    pub unsafe fn from_raw(ptr: *mut DLManagedTensor) -> Self {
        debug_assert!(!ptr.is_null());
        let proxy = (*ptr).into();
        ManagedTensor {
            inner: proxy,
            _marker: PhantomData,
        }
    }

    /// Consumes the ManagedTensor and returns Tensor.
    pub fn into_tensor(self) -> Tensor<'tensor> {
        self.inner.dl_tensor.into()
    }
}
