#include <ros/ros.h>
#include <exploration_manager/fast_exploration_fsm.h>

#include <plan_manage/backward.hpp>

// 创建一个 FastExplorationFSM 对象，调它的 init()，然后 ros::spin() 等回调。容器里的 exploration_node 进程跑的就是它

namespace backward {
backward::SignalHandling sh;
}

using namespace fast_planner;

int main(int argc, char** argv) {
  ros::init(argc, argv, "exploration_node");
  ros::NodeHandle nh("~");

  FastExplorationFSM expl_fsm;
  expl_fsm.init(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
