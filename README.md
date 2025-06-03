# From Detection to Explanation: Effective Learning Strategies for LLMs in Online Abusive Language Research

[![Preprint](https://img.shields.io/badge/Preprint-arXiv-red)](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/316198577/2025_COLING_from_detection_to_explanation.pdf)

**TL-DR:** In this paper, we (1) demonstrate that LLM's implicit knowledge without an accurate learning strategy does not effectively capture multi-class abusive language; (2) we propose an easy and open source knowledge-guided learning strategy to evaluate the impact of adding external knowledge to LLMs; (3) building on this strategy, we release GLlama Alarm, a knowledge-guided version of Llama-2 instruction fine-tuned for multi-class abusive language detection and explanation generation; (4) we conduct an expert survey to evaluate LLMs in abusive language research, from detection to explanation.

**Authors:** [Chiara Di Bonaventura](https://kclpure.kcl.ac.uk/portal/en/persons/chiara-di-bonaventura), [Lucia Siciliani](https://swap.di.uniba.it/members/siciliani.lucia/), [Pierpaolo Basile](https://swap.di.uniba.it/members/basile.pierpaolo/), [Albert Meroño-Peñuela](https://www.albertmeronyo.org/), [Barbara McGillivray](https://www.kcl.ac.uk/people/barbara-mcgillivray)

**Contact:** [chiara.di_bonaventura@kcl.ac.uk](mailto:chiara.di_bonaventura@kcl.ac.uk)

## 📂 Repo info

This repository contains the code used to instruction-finetuning Llama-2. 

## 📎 Citation 
```bibtex
@inproceedings{di-bonaventura-etal-2025-detection,
    title = "From Detection to Explanation: Effective Learning Strategies for {LLM}s in Online Abusive Language Research",
    author = "Di Bonaventura, Chiara  and
      Siciliani, Lucia  and
      Basile, Pierpaolo  and
      Merono Penuela, Albert  and
      McGillivray, Barbara",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.141/",
    pages = "2067--2084",
    abstract = "Abusive language detection relies on understanding different levels of intensity, expressiveness and targeted groups, which requires commonsense reasoning, world knowledge and linguistic nuances that evolve over time. Here, we frame the problem as a knowledge-guided learning task, and demonstrate that LLMs' implicit knowledge without an accurate strategy is not suitable for multi-class detection nor explanation generation. We publicly release GLlama Alarm, the knowledge-Guided version of Llama-2 instruction fine-tuned for multi-class abusive language detection and explanation generation. By being fine-tuned on structured explanations and external reliable knowledge sources, our model mitigates bias and generates explanations that are relevant to the text and coherent with human reasoning, with an average 48.76{\%} better alignment with human judgment according to our expert survey."
}
```

## 📌 License
This project is licensed under the CC-BY-4.0 License.
