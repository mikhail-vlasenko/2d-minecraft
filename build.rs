use config_struct::{DynamicLoading, Error, FloatSize, IntSize, SerdeSupport, StructOptions};


fn main() -> Result<(), Error> {
    let options = StructOptions {
        format: None,
        struct_name: "Settings".to_owned(),
        const_name: Some("DEFAULT_SETTINGS".to_owned()),
        generate_const: true,
        derived_traits: vec!["Debug".to_owned(), "Clone".to_owned()],
        serde_support: SerdeSupport::Yes,
        generate_load_fns: true,
        use_serde_derive_crate: false,
        dynamic_loading: DynamicLoading::DebugOnly,
        create_dirs: true,
        write_only_if_changed: true,
        default_float_size: FloatSize::F32,
        default_int_size: IntSize::I32,
        max_array_size: 0,
    };
    config_struct::create_struct(
        "settings.yaml",
        "src/settings.rs",
        &options)
}
