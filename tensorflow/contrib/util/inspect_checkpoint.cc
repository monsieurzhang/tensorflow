/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

#include "tensorflow/c/checkpoint_reader.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

namespace tensorflow {
namespace {

int LoadCheckpoint(const string& in)
{
  // set up your input paths
  const string pathToGraph = "model.ckpt-1000000.meta";
  const string checkpointPath = in;

  auto session = NewSession(SessionOptions());
  if (session == nullptr) {
    throw std::runtime_error("Could not create Tensorflow session.");
  }

  std::cerr << "gl" << std::endl;

  Status status;

  // Read in the protobuf graph we exported
  MetaGraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
  if (!status.ok()) {
    throw std::runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
  }

  // Add the graph to the session
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
    throw std::runtime_error("Error creating graph: " + status.ToString());
  }

  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkpointPath;
  status = session->Run(
  { { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, },
  {},
  { graph_def.saver_def().restore_op_name() },
    nullptr);
  if (!status.ok()) {
    throw std::runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
  }

  // and run the inference to your liking
/*  auto feedDict = ...
    auto outputOps = ...
    std::vector<tensorflow::Tensor> outputTensors;
  status = session->Run(feedDict, outputOps, {}, &outputTensors);
*/
  return 0;
}

int InspectCheckpoint(const string& in) 
{
/*
  tensorflow::checkpoint::TensorSliceReader reader(
      in, tensorflow::checkpoint::OpenTableTensorSliceReader);
  Status s = reader.status();
  if (!s.ok()) {
    fprintf(stderr, "Unable to open the checkpoint file\n");
    return -1;
  }
  for (auto e : reader.Tensors()) {
    fprintf(stdout, "%s %s\n", e.first.c_str(),
            e.second->shape().DebugString().c_str());
  }
*/
  checkpoint::CheckpointReader *result = 0;
  TF_Status *arg2 = (TF_Status *)0;

  result = (checkpoint::CheckpointReader *)new checkpoint::CheckpointReader((string const &)in, arg2);

  auto var_to_shape_map = result->GetVariableToShapeMap();
  for (auto a : var_to_shape_map)
  {
    std::cerr << a.first << std::endl;

    std::unique_ptr<tensorflow::Tensor> out_tensor;
    TF_Status *tf_s = (TF_Status *)0;

    result->GetTensor(a.first, &out_tensor, tf_s);

    StringPiece sp_temp = out_tensor.get()->tensor_data();

    const char* data_temp = sp_temp.data();
    size_t s_temp = sp_temp.size();

    std::cerr << out_tensor.get()->dtype() << std::endl;
  }



  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 2) {
    fprintf(stderr, "Usage: %s checkpoint_file\n", argv[0]);
    exit(1);
  }
//  return tensorflow::InspectCheckpoint(argv[1]);
  return tensorflow::LoadCheckpoint(argv[1]);
}
