"""Pilot-v2 clean manifold：仅用来源攻击 Train clean 拟合。"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


class CleanManifold:
    """对应 STEP 8/10：逐层逐池化 PCA + 收缩协方差的 clean reference。

    每个参考对象属于单一来源攻击；跨攻击测试仍使用该来源对象，禁止替换为目标参考。
    """

    def __init__(self, dimension: int, seed: int):
        """输入：dimension 为 PCA 维度；seed 控制 randomized SVD。
        输出：未拟合的参考对象，不读取数据、不计算分数。
        """
        self.dimension, self.seed = dimension, seed

    def fit(self, clean_vectors: np.ndarray):
        """对应 STEP 8：仅接收来源攻击 Train clean 的层级池化向量拟合 PCA/协方差。

        输入：形状 (clean样本数,层数,2,hidden_size)，样本数应大于 PCA 维度。
        输出：self；每个层/池化保存 PCA 和 LedoitWolf，其他 split 不参与拟合。
        """
        self.models = []
        flat = clean_vectors.reshape(len(clean_vectors), -1, clean_vectors.shape[-1])
        for index in range(flat.shape[1]):
            pca = PCA(n_components=self.dimension, svd_solver="randomized", random_state=self.seed)
            projected = pca.fit_transform(flat[:, index].astype(np.float32))
            covariance = LedoitWolf().fit(projected)
            self.models.append((pca, covariance))
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """对应 STEP 8：用已冻结 reference 计算 Mahalanobis 距离的均值和最大值。

        公式：d=sqrt((z−μ)^T precision (z−μ))，z 是逐层逐池化 PCA 投影。
        输入：与 fit 相同层/池化顺序的任意 split 向量；不重新拟合。
        输出：(样本数,2) 数组，列为 mahalanobis_mean / mahalanobis_max。
        """
        flat = vectors.reshape(len(vectors), -1, vectors.shape[-1])
        distances = []
        for index, (pca, covariance) in enumerate(self.models):
            projected = pca.transform(flat[:, index].astype(np.float32))
            distances.append(np.sqrt(np.maximum(covariance.mahalanobis(projected), 0)))
        values = np.asarray(distances).T
        return np.column_stack([values.mean(axis=1), values.max(axis=1)])
