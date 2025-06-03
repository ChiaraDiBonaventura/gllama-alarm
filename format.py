from abc import ABC, abstractmethod


class Format(ABC):

    def __init__(self, instruction_field: str, response_field: str):

        self.instruction_field = instruction_field
        self.response_field = response_field

    @abstractmethod
    def get_prompt(self, example: dict, is_train: bool) -> str:
        raise NotImplementedError


class AbusiveAlpacaFormat(Format):

    format_name = "abusive_alpaca"

    def __init__(self, response_field: str, dataset_field:str, adding_context: bool, context_field: str = None, kg_context_field:str = None):

        super().__init__(kg_context_field, response_field)
        self.dataset = dataset_field
        self.context_field = context_field
        self.adding_context = adding_context


    def get_prompt(self, example: dict, is_train: bool) -> str:

        response = example[self.response_field] if is_train else ""
        context = example.get(self.context_field, "") if self.context_field else ""

        # check which dataset
        if self.dataset == 'hatexplain': 
            instruction = f"Classify the input text as 'hate speech', 'offensive' or 'neutral'."
        elif self.dataset == 'implicit_hate': 
            instruction = "Classify the input text as 'implicit hate speech', 'explicit hate speech' or 'neutral'."

        kg_context = example[self.instruction_field]

        if kg_context != "" and self.adding_context == True:
            instruction += f"\n\nContext: {example[self.instruction_field]}."

        # Instruction-tuning conditional with KG and without
        if self.adding_context == True:

            return "Below is an instruction that describes a task, paired with context and input text. " \
                "Write a response that appropriately completes the instruction based on the context.\n\n" \
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Answer:\n{response}"

        else:

            return "Below is an instruction that describes a task, paired with an input text. " \
                f"Write a response that appropriately completes the instruction.\n\n" \
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Answer:\n{response}"


class AlpacaFormat(Format):

    format_name = "alpaca"

    def __init__(self, instruction_field: str, response_field: str, dataset_field:str, context_field: str = None, kg_context_field:str = None):

        super().__init__(instruction_field, response_field)
        self.dataset = dataset_field
        self.context_field = context_field
        self.kg_context_field = kg_context_field

    def get_prompt(self, example: dict, is_train: bool) -> str:

        instruction = example[self.instruction_field]
        response = example[self.response_field] if is_train else ""
        context = example.get(self.context_field, "") if self.context_field else ""
        kg_context = example.get(self.kg_context_field, "") if self.context_field else ""

        # check which dataset
        if self.dataset == 'hatexplain': 
            instruction = f"Classify the input text as 'hate speech', 'offensive' or 'neutral'.\n\nContext: {kg_context}." 
        elif self.dataset == 'implicit_hate': 
            instruction = "Classify the input text as 'implicit hate speech', 'explicit hate speech' or 'not hate speech'.\n\nContext: {kg_context}."

        # modificare questo pezzo per fare tuning con e senza KG
        if kg_context != "":

            return "Below is an instruction that describes a task, paired with context and input text. " \
                "Write a response that appropriately completes the instruction based on the context.\n\n" \
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Answer:\n{response}"

        else:

            return "Below is an instruction that describes a task, paired with an input text. " \
                f"Write a response that appropriately completes the instruction.\n\n" \
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Answer:\n{response}"
