// install the tensorflow lite library from https://github.com/bsatrom/Particle_TensorFlowLite/blob/master/README.md#installation

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "linear_regression_model_data.cpp"

const tflite::Model *model = tflite::GetModel(g_linear_regresion_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION)
{
  error_reporter->Report(
    "Model provided is schema version %d not equal "
    "to supported version %d.",
    model->version(), TFLITE_SCHEMA_VERSION);
  return;
}

//Model invoke

float x_val = randFloat(0, 1);
input->data.f[0] = x_val;

TfLiteStatus invoke_status = interpreter->Invoke();
if (invoke_status != kTfLiteOk)
{
  error_reporter->Report("Invoke failed on x_val: %f\n",
                          static_cast<double>(x_val));
  return;
}

float y_val = output->data.f[0];
