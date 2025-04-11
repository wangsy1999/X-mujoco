# X-mujoco
本仿真环境适配了天工X-humanoid lite，可以用于测试训练结果。

此环境基于Dknt的hhfc-mujoco进行修改，使用bitbot-frontend进行控制。

环境依赖：libtorch。 

**注意：**
libtorch需要下载cxx11 ABI的版本，cuda和cpu版均可。https://pytorch.org/get-started/locally/


## 使用方法
首先在isaacgym进行play

将log中的exported内的policy_1.pt 复制到X-mujoco/checkpoint

在X-mujoco文件夹内

cd build

cmake .. 

make   

./bin/main_app

打开bitbot-frontend，连接，点击控制，依次快速按下8，p，r，如果机器人倒地可在mujoco中点击reset 

bitbot-frontend的github仓库：https://github.com/limymy/bitbot_frontend-release/releases

## 已知问题
目前只能读取ActorCritic的policy_class导出的的pt文件，没有适配lstm,因此无法读取ActorCriticRecurrent

用cpu推理可能出现速度缓慢的情况（是cpu版libtorch的共同问题）



