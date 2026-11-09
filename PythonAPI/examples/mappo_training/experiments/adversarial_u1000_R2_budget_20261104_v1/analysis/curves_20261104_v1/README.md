# R2曲线与TensorBoard回放

本轮没有原生TensorBoard事件文件。此目录从已完成的JSONL/冻结评估日志派生，原始数据未修改。训练和冻结评估使用独立run，横轴为A+B实际train ticks。

图中深色线为尾随12轮均值，TensorBoard保存未平滑逐轮值。training每队128点、frozen每队4点，已重新读取事件文件核验。

在mappo_training工作目录运行：

```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m tensorboard.main --logdir "D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training\experiments\adversarial_u1000_R2_budget_20261104_v1\analysis\curves_20261104_v1\tensorboard_replay" --port 6006
```
