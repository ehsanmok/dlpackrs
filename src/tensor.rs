use pin_project::{pin_project, pinned_drop};

use core::slice;
use std::{
    fmt::Debug,
    marker::{PhantomData, PhantomPinned},
    mem::transmute,
    os::raw::c_void,
    pin::Pin,
    ptr::{self, NonNull},
};

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

/// A typed ManagerContext type that is `!Unpin` i.e. pinnable for safety since it holds a pointer to the underlying DLTensor.
#[derive(Debug)]
#[repr(C)]
pub struct ManagerContext<C> {
    pub ptr: Option<NonNull<*mut c_void>>,
    ty: PhantomData<C>,
    _pin: PhantomPinned,
}

impl<C> ManagerContext<C> {
    pub fn new(ptr: Option<NonNull<*mut c_void>>) -> Self {
        Self {
            ptr,
            ty: PhantomData,
            _pin: PhantomPinned,
        }
    }
}

/// Safe proxy to ffi::DLManagedTensor which is self-referential by design.
/// See [DLManagedTensor](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv415DLManagedTensor)
#[pin_project(PinnedDrop)]
#[repr(C)]
pub struct ManagedTensorProxy<C> {
    /// Holds the underlying tensor.
    pub dl_tensor: DLTensor,
    /// The context holding the underlying DLTensor.
    #[pin]
    pub manager_ctx: ManagerContext<C>, // safe typed wrapper for *mut c_void which is !Unpin i.e. pinnable
    /// Deleter function pointer.
    // TODO: should this be `#[pin]`?
    pub deleter: Option<fn(&mut ManagedTensor<C>)>,
}

impl<C: Debug> Debug for ManagedTensorProxy<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedTensorProxy")
            .field("dl_tensor", &self.dl_tensor)
            .field("manager_ctx", &self.manager_ctx)
            .finish()
    }
}

impl<C> ManagedTensorProxy<C> {
    pub fn dl_tensor(&self) -> DLTensor {
        self.dl_tensor
    }

    pub fn manager_ctx(self: Pin<&mut Self>) -> Option<NonNull<*mut c_void>> {
        let mut this = self.project();
        this.manager_ctx.as_mut().ptr
    }

    pub fn set_manager_ctx(self: Pin<&mut Self>, manager_ctx: NonNull<*mut c_void>) {
        let mut this = self.project();
        let new = ManagerContext::new(Some(manager_ctx));
        this.manager_ctx.set(new);
    }
}

impl<C> From<DLManagedTensor> for ManagedTensorProxy<C> {
    fn from(mut dlmt: DLManagedTensor) -> Self {
        let ptr: Option<NonNull<*mut c_void>> = if dlmt.manager_ctx.is_null() {
            None
        } else {
            unsafe { Some(NonNull::new_unchecked(&mut dlmt.manager_ctx as *mut _)) }
        };
        let manager_ctx = ManagerContext::new(ptr);
        let deleter = dlmt.deleter.take().map(|del| unsafe {
            transmute::<unsafe extern "C" fn(*mut DLManagedTensor), fn(&mut ManagedTensor<C>)>(del)
        });
        ManagedTensorProxy {
            dl_tensor: dlmt.dl_tensor,
            manager_ctx,
            deleter,
        }
    }
}

impl<C> From<ManagedTensorProxy<C>> for DLManagedTensor {
    fn from(pmt: ManagedTensorProxy<C>) -> Self {
        let dl_tensor = pmt.dl_tensor;
        let manager_ctx = match pmt.manager_ctx.ptr {
            None => ptr::null_mut(),
            Some(nnptr) => unsafe { *nnptr.as_ptr() },
        };
        let deleter = unsafe {
            pmt.deleter.map(|del_fn| {
                transmute::<fn(&mut ManagedTensor<C>), unsafe extern "C" fn(*mut DLManagedTensor)>(
                    del_fn,
                )
            })
        };
        DLManagedTensor {
            dl_tensor,
            manager_ctx,
            deleter,
        }
    }
}

impl<C> From<Pin<&mut ManagedTensorProxy<C>>> for DLManagedTensor {
    fn from(pmt: Pin<&mut ManagedTensorProxy<C>>) -> Self {
        let dl_tensor = pmt.dl_tensor;
        let manager_ctx = match pmt.manager_ctx.ptr {
            None => ptr::null_mut(),
            Some(nnptr) => unsafe { *nnptr.as_ptr() },
        };
        let deleter = unsafe {
            pmt.deleter.map(|del_fn| {
                transmute::<fn(&mut ManagedTensor<C>), unsafe extern "C" fn(*mut DLManagedTensor)>(
                    del_fn,
                )
            })
        };
        DLManagedTensor {
            dl_tensor,
            manager_ctx,
            deleter,
        }
    }
}

#[allow(clippy::needless_lifetimes)]
#[pinned_drop]
impl<C> PinnedDrop for ManagedTensorProxy<C> {
    fn drop(mut self: Pin<&mut Self>) {
        let mut dlm: DLManagedTensor = self.as_mut().into();
        if let Some(fptr) = self.deleter {
            unsafe {
                let cfptr = transmute::<fn(&mut ManagedTensor<C>), fn(*mut DLManagedTensor)>(fptr);
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
pub struct ManagedTensor<'tensor, C: 'tensor> {
    pub inner: ManagedTensorProxy<C>,
    _marker: PhantomData<fn(&'tensor ()) -> &'tensor ()>, // invariant wrt 'tensor
}

impl<'tensor, C> From<DLManagedTensor> for ManagedTensor<'tensor, C> {
    fn from(dlm: DLManagedTensor) -> Self {
        let proxy: ManagedTensorProxy<C> = dlm.into();
        ManagedTensor {
            inner: proxy,
            _marker: PhantomData,
        }
    }
}

impl<'tensor, C> From<ManagedTensor<'tensor, C>> for DLManagedTensor {
    fn from(mt: ManagedTensor<'tensor, C>) -> Self {
        mt.inner.into()
    }
}

impl<'tensor, C: 'tensor> ManagedTensor<'tensor, C> {
    /// Contructor.
    pub fn new(tensor: Tensor<'tensor>, manager_ctx: Option<NonNull<*mut c_void>>) -> Self {
        let manager_ctx = ManagerContext::new(manager_ctx);
        let inner = ManagedTensorProxy {
            dl_tensor: tensor.into_inner(),
            manager_ctx,
            deleter: None,
        };

        ManagedTensor {
            inner,
            _marker: PhantomData,
        }
    }

    /// Sets a deleter function pointer.
    pub fn set_deleter(&mut self, deleter: fn(&mut ManagedTensor<C>)) {
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
        ManagedTensor {
            inner: (*ptr).into(),
            _marker: PhantomData,
        }
    }

    /// Consumes the ManagedTensor and returns Tensor.
    pub fn into_tensor(self) -> Tensor<'tensor> {
        self.inner.dl_tensor.into()
    }
}
