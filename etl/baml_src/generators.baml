// BAML Generator Configuration
// Generates typed Python client from BAML definitions.
// Output goes to etl/baml_client/ adjacent to etl/baml_src/.

generator target {
  output_type "python/pydantic"
  output_dir "../"
  version "0.220.0"
  default_client_mode "sync"
}
