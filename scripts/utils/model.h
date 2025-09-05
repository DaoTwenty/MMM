#pragma once

#include <string>

#include <onnxruntime_cxx_api.h>

namespace mmm {

class Model {
public:

Model(const std::string& modelFilepath);
~Model();

void initialize();
void forward();

private:
  std::shared_ptr<Ort::Env> m_Env;
  std::shared_ptr<Ort::Session> m_Session;
  char* m_InputName;
  std::vector<int64_t> m_InputDims;
  char* m_OutputName;
  std::vector<int64_t> m_OutputDims;

}

}