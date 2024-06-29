#include "qwen2_model.h"
#include "getmem.h"

int main(int argc, char **argv) {
    std::string modelpath = "Qwen1.5-1.8B-Chat-GPTQ-Int4-lite";
    std::string user_prompt = "Hello! How are you?";

    if (argc > 1) {
        modelpath = argv[1];
    }
    if (argc > 2) {
        user_prompt = argv[2];
    }

    GPT2Tokenizer tokenizer = GPT2Tokenizer::load(modelpath+"/vocab.json", modelpath+"/merges.txt");

    Model model(modelpath);

    std::vector<std::string> chat_template = {"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n","<|im_end|>\n<|im_start|>assistant\n"};
    user_prompt = chat_template[0] + user_prompt + chat_template[1];

    std::vector<int> input_ids = tokenizer.encode_template(user_prompt);
    
    std::string output = model.generate(input_ids,tokenizer,1024,false,true,true);
    printf("CurrRSS: %zuM & PeakRSS: %zuM\n", getCurrentRSS()>>20, getPeakRSS()>>20);

    // model.benchmark(tokenizer,128);

    model.clear();

    return EXIT_SUCCESS;
}