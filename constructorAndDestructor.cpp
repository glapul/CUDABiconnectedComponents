BiconnectedComponents::BiconnectedComponents() {
    cuInit(0);
    _cu_exec(cuDeviceGet(&_cu_device, 0));
    _cu_exec(cuCtxCreate(&_cu_context, 0, _cu_device));
    _cu_exec(cuModuleLoad(&_cu_module, "BiconnectedComponents.ptx"));
}
BiconnectedComponents::~BiconnectedComponents() {

}

