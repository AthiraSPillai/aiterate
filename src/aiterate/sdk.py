from aiterate.domain import OptimizationRequest
from aiterate.optimizer import SkillOptInspiredOptimizer


class AIterateClient:
    def __init__(self) -> None:
        self.optimizer = SkillOptInspiredOptimizer()

    def optimize(self, request: OptimizationRequest):
        return self.optimizer.optimize(request)

