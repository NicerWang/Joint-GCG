from dataclasses import dataclass

from attack_rag import attack_base as attack_base_rag
from attack_rag import attack_joint as attack_joint_rag
from ppl_defense import attack_joint_w_ppl_filter


@dataclass
class PoisionedRAGJoint:
    method: callable = attack_joint_rag
    tag: str = "rag_v2"
    use_adaptive_ratio: bool = True


@dataclass
class PoisionedRAGBaseline:
    method: callable = attack_joint_rag
    tag: str = "rag_base"
    use_adaptive_ratio: bool = False


@dataclass
class LIARRAGBaseline:
    method: callable = attack_base_rag
    tag: str = "rag_k"


@dataclass
class PoisionedRAGAblation1:
    method: callable = attack_joint_rag
    tag: str = "rag_ablation1"
    use_adaptive_ratio: bool = False


@dataclass
class PoisionedRAGAblation2:
    method: callable = attack_joint_rag
    tag: str = "rag_ablation2"
    use_adaptive_ratio: bool = True
    joint_loss_only: bool = True


@dataclass
class PoisionedRAGAblation3:
    method: callable = attack_joint_rag
    tag: str = "rag_ablation3"
    use_adaptive_ratio: bool = True
    joint_grad_only: bool = True


@dataclass
class PoisionedRAGDefensePPL:
    method: callable = attack_joint_w_ppl_filter
    tag: str = "rag_ppl8"
    use_adaptive_ratio: bool = True
    tag_length: int = 16
