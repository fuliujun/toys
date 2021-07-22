import TensorRT as trt
import sys
def build_engine(onnx_file_path,engine_save_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with builder.create_builder_config() as config:
            config.max_workspace_size = 1 << 20
            config.max_batch_size = 1
        with open(onnx_file_path,'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            else:
                engine = builder.build_engine(network, config)
                with open(engine_save_path, 'wb') as f:
                    f.write(engine.serialize())
                

if __name__ == "__main__":
    onnx_path_file = sys.argv[1]
    engine_path_file = sys.argv[2]
    build_engine(onnx_path_file,engine_path_file)
    # inputs = np.random.random((1, 3, 1024, 853)).astype(np.float32)
    
