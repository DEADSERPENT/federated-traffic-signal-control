"""
Python-based Edge/Cloud Simulation
Mimics CloudSim Plus functionality for traffic signal FL system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import time


@dataclass
class ProcessingElement:
    """Represents a CPU processing element (PE)"""
    pe_id: int
    mips: int  # Million Instructions Per Second
    utilization: float = 0.0


@dataclass
class VirtualMachine:
    """Represents a Virtual Machine"""
    vm_id: int
    mips: int
    ram: int  # MB
    bandwidth: int  # Mbps
    size: int  # Storage in MB
    pes: int = 1
    description: str = ""

    # Runtime metrics
    total_execution_time: float = 0.0
    tasks_completed: int = 0


@dataclass
class Cloudlet:
    """Represents a computational task (cloudlet)"""
    cloudlet_id: int
    length: int  # Million Instructions (MI)
    pes_required: int = 1

    # Execution tracking
    vm_id: int = -1
    start_time: float = 0.0
    finish_time: float = 0.0
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED

    @property
    def execution_time(self) -> float:
        if self.status == "COMPLETED":
            return self.finish_time - self.start_time
        return 0.0


@dataclass
class Host:
    """Represents a physical host machine"""
    host_id: int
    mips: int
    ram: int
    bandwidth: int
    storage: int
    pes: List[ProcessingElement] = field(default_factory=list)
    vms: List[VirtualMachine] = field(default_factory=list)


@dataclass
class Datacenter:
    """Represents a datacenter (edge or cloud)"""
    name: str
    hosts: List[Host] = field(default_factory=list)

    @property
    def total_mips(self) -> int:
        return sum(h.mips * len(h.pes) for h in self.hosts)

    @property
    def total_ram(self) -> int:
        return sum(h.ram for h in self.hosts)


class EdgeCloudSimulator:
    """
    Simulates edge and cloud computing infrastructure for
    federated learning-based traffic signal control.
    """

    # Configuration
    NUM_EDGE_NODES = 4
    EDGE_VM_MIPS = 1000
    EDGE_VM_RAM = 512
    EDGE_VM_BW = 1000
    EDGE_VM_SIZE = 10000

    CLOUD_HOSTS = 4
    CLOUD_HOST_MIPS = 10000
    CLOUD_HOST_RAM = 8192
    CLOUD_HOST_BW = 10000
    CLOUD_HOST_STORAGE = 1000000
    CLOUD_HOST_PES = 8

    LOCAL_TRAINING_LENGTH = 10000  # MI
    AGGREGATION_LENGTH = 5000  # MI

    def __init__(self, config: Dict = None):
        """Initialize the simulator."""
        self.config = config or {}
        self.simulation_time = 0.0

        # Create infrastructure
        self.edge_datacenter = self._create_edge_datacenter()
        self.cloud_datacenter = self._create_cloud_datacenter()

        # Create VMs
        self.edge_vms = self._create_edge_vms()
        self.cloud_vms = self._create_cloud_vms()

        # Create cloudlets
        self.edge_cloudlets: List[Cloudlet] = []
        self.cloud_cloudlets: List[Cloudlet] = []

        # Metrics
        self.metrics_history: List[Dict] = []

    def _create_edge_datacenter(self) -> Datacenter:
        """Create edge datacenter with limited resources."""
        hosts = []
        for i in range(self.NUM_EDGE_NODES):
            pes = [ProcessingElement(pe_id=0, mips=self.EDGE_VM_MIPS)]
            host = Host(
                host_id=i,
                mips=self.EDGE_VM_MIPS,
                ram=self.EDGE_VM_RAM,
                bandwidth=self.EDGE_VM_BW,
                storage=self.EDGE_VM_SIZE,
                pes=pes
            )
            hosts.append(host)

        return Datacenter(name="EdgeDatacenter", hosts=hosts)

    def _create_cloud_datacenter(self) -> Datacenter:
        """Create cloud datacenter with high resources."""
        hosts = []
        for i in range(self.CLOUD_HOSTS):
            pes = [
                ProcessingElement(pe_id=j, mips=self.CLOUD_HOST_MIPS)
                for j in range(self.CLOUD_HOST_PES)
            ]
            host = Host(
                host_id=i,
                mips=self.CLOUD_HOST_MIPS,
                ram=self.CLOUD_HOST_RAM,
                bandwidth=self.CLOUD_HOST_BW,
                storage=self.CLOUD_HOST_STORAGE,
                pes=pes
            )
            hosts.append(host)

        return Datacenter(name="CloudDatacenter", hosts=hosts)

    def _create_edge_vms(self) -> List[VirtualMachine]:
        """Create VMs for edge nodes."""
        vms = []
        for i in range(self.NUM_EDGE_NODES):
            vm = VirtualMachine(
                vm_id=i,
                mips=self.EDGE_VM_MIPS,
                ram=self.EDGE_VM_RAM,
                bandwidth=self.EDGE_VM_BW,
                size=self.EDGE_VM_SIZE,
                pes=1,
                description=f"EdgeNode_{i}"
            )
            vms.append(vm)
        return vms

    def _create_cloud_vms(self) -> List[VirtualMachine]:
        """Create VMs for cloud aggregation."""
        vm = VirtualMachine(
            vm_id=0,
            mips=self.CLOUD_HOST_MIPS,
            ram=self.CLOUD_HOST_RAM,
            bandwidth=self.CLOUD_HOST_BW,
            size=self.CLOUD_HOST_STORAGE,
            pes=self.CLOUD_HOST_PES,
            description="CloudAggregator"
        )
        return [vm]

    def create_local_training_cloudlets(self, num_rounds: int = 5) -> List[Cloudlet]:
        """Create cloudlets for local ML training at edge nodes."""
        cloudlets = []
        cloudlet_id = 0

        for i in range(self.NUM_EDGE_NODES):
            for _ in range(num_rounds):
                cloudlet = Cloudlet(
                    cloudlet_id=cloudlet_id,
                    length=self.LOCAL_TRAINING_LENGTH,
                    pes_required=1
                )
                cloudlets.append(cloudlet)
                cloudlet_id += 1

        self.edge_cloudlets = cloudlets
        return cloudlets

    def create_aggregation_cloudlets(self, num_rounds: int = 5) -> List[Cloudlet]:
        """Create cloudlets for global model aggregation."""
        cloudlets = []

        for i in range(num_rounds):
            cloudlet = Cloudlet(
                cloudlet_id=i,
                length=self.AGGREGATION_LENGTH,
                pes_required=1
            )
            cloudlets.append(cloudlet)

        self.cloud_cloudlets = cloudlets
        return cloudlets

    def execute_cloudlet(self, cloudlet: Cloudlet, vm: VirtualMachine) -> float:
        """
        Execute a cloudlet on a VM and return execution time.

        Execution time = Length (MI) / MIPS
        """
        # Calculate execution time
        execution_time = cloudlet.length / vm.mips

        # Add some variance for realism
        execution_time *= np.random.uniform(0.9, 1.1)

        # Update cloudlet
        cloudlet.vm_id = vm.vm_id
        cloudlet.start_time = self.simulation_time
        cloudlet.finish_time = self.simulation_time + execution_time
        cloudlet.status = "COMPLETED"

        # Update VM metrics
        vm.total_execution_time += execution_time
        vm.tasks_completed += 1

        # Advance simulation time
        self.simulation_time = cloudlet.finish_time

        return execution_time

    def run_simulation(self, num_fl_rounds: int = 5) -> Dict[str, Any]:
        """
        Run the complete edge/cloud simulation.

        Args:
            num_fl_rounds: Number of federated learning rounds to simulate

        Returns:
            Simulation results dictionary
        """
        print("=" * 60)
        print("Edge/Cloud Simulation for Traffic Signal FL System")
        print("=" * 60)

        # Create cloudlets
        self.create_local_training_cloudlets(num_fl_rounds)
        self.create_aggregation_cloudlets(num_fl_rounds)

        print(f"\nSimulation Configuration:")
        print(f"  - Edge Nodes: {self.NUM_EDGE_NODES}")
        print(f"  - Cloud Hosts: {self.CLOUD_HOSTS}")
        print(f"  - FL Rounds: {num_fl_rounds}")
        print(f"  - Edge Cloudlets: {len(self.edge_cloudlets)}")
        print(f"  - Cloud Cloudlets: {len(self.cloud_cloudlets)}")

        # Execute edge cloudlets (local training)
        print("\n" + "-" * 50)
        print("Executing Edge Node Tasks (Local Training)")
        print("-" * 50)
        print(f"{'Cloudlet':<10} {'VM':<10} {'Start':<12} {'Finish':<12} {'ExecTime':<12}")
        print("-" * 56)

        edge_results = []
        cloudlet_idx = 0

        for round_num in range(num_fl_rounds):
            for vm in self.edge_vms:
                cloudlet = self.edge_cloudlets[cloudlet_idx]
                exec_time = self.execute_cloudlet(cloudlet, vm)

                print(f"{cloudlet.cloudlet_id:<10} {vm.vm_id:<10} "
                      f"{cloudlet.start_time:<12.2f} {cloudlet.finish_time:<12.2f} "
                      f"{exec_time:<12.2f}")

                edge_results.append({
                    "cloudlet_id": cloudlet.cloudlet_id,
                    "vm_id": vm.vm_id,
                    "round": round_num,
                    "start_time": cloudlet.start_time,
                    "finish_time": cloudlet.finish_time,
                    "execution_time": exec_time
                })

                cloudlet_idx += 1

        # Execute cloud cloudlets (aggregation)
        print("\n" + "-" * 50)
        print("Executing Cloud Tasks (Global Aggregation)")
        print("-" * 50)
        print(f"{'Cloudlet':<10} {'VM':<10} {'Start':<12} {'Finish':<12} {'ExecTime':<12}")
        print("-" * 56)

        cloud_results = []
        cloud_vm = self.cloud_vms[0]

        for cloudlet in self.cloud_cloudlets:
            exec_time = self.execute_cloudlet(cloudlet, cloud_vm)

            print(f"{cloudlet.cloudlet_id:<10} {cloud_vm.vm_id:<10} "
                  f"{cloudlet.start_time:<12.2f} {cloudlet.finish_time:<12.2f} "
                  f"{exec_time:<12.2f}")

            cloud_results.append({
                "cloudlet_id": cloudlet.cloudlet_id,
                "vm_id": cloud_vm.vm_id,
                "start_time": cloudlet.start_time,
                "finish_time": cloudlet.finish_time,
                "execution_time": exec_time
            })

        # Calculate summary
        total_edge_time = sum(r["execution_time"] for r in edge_results)
        total_cloud_time = sum(r["execution_time"] for r in cloud_results)

        # Print summary
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"\nEdge Computing Metrics:")
        print(f"  - Total Edge Cloudlets: {len(edge_results)}")
        print(f"  - Total Edge Execution Time: {total_edge_time:.2f} seconds")
        print(f"  - Average Task Time: {total_edge_time/len(edge_results):.2f} seconds")

        for vm in self.edge_vms:
            print(f"  - {vm.description}: {vm.tasks_completed} tasks, "
                  f"{vm.total_execution_time:.2f}s total")

        print(f"\nCloud Computing Metrics:")
        print(f"  - Total Cloud Cloudlets: {len(cloud_results)}")
        print(f"  - Total Cloud Execution Time: {total_cloud_time:.2f} seconds")
        print(f"  - Average Task Time: {total_cloud_time/len(cloud_results):.2f} seconds")
        print(f"  - {cloud_vm.description}: {cloud_vm.tasks_completed} tasks")

        print(f"\nResource Utilization:")
        print(f"  - Edge Datacenter Total MIPS: {self.edge_datacenter.total_mips}")
        print(f"  - Cloud Datacenter Total MIPS: {self.cloud_datacenter.total_mips}")
        print(f"  - Simulation End Time: {self.simulation_time:.2f} seconds")

        print("=" * 60)

        return {
            "edge_results": edge_results,
            "cloud_results": cloud_results,
            "total_edge_time": total_edge_time,
            "total_cloud_time": total_cloud_time,
            "simulation_end_time": self.simulation_time,
            "edge_vms": [{"vm_id": vm.vm_id, "tasks": vm.tasks_completed,
                         "total_time": vm.total_execution_time} for vm in self.edge_vms],
            "cloud_vms": [{"vm_id": vm.vm_id, "tasks": vm.tasks_completed,
                         "total_time": vm.total_execution_time} for vm in self.cloud_vms]
        }


def main():
    """Run the edge/cloud simulation."""
    simulator = EdgeCloudSimulator()
    results = simulator.run_simulation(num_fl_rounds=5)
    return results


if __name__ == "__main__":
    main()
