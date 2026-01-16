package traffic;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

/**
 * CloudSim Plus simulation for Traffic Signal Control System.
 *
 * This simulation models:
 * - Edge nodes (traffic intersections) as VMs with limited resources
 * - Cloud datacenter for federated aggregation
 * - Computational tasks for local ML training and global aggregation
 */
public class TrafficEdgeCloudSimulation {

    // Edge node configuration
    private static final int NUM_EDGE_NODES = 4;
    private static final int EDGE_VM_MIPS = 1000;
    private static final int EDGE_VM_RAM = 512;      // MB
    private static final int EDGE_VM_BW = 1000;      // Mbps
    private static final int EDGE_VM_SIZE = 10000;   // MB storage

    // Cloud datacenter configuration
    private static final int CLOUD_HOSTS = 4;
    private static final int CLOUD_HOST_MIPS = 10000;
    private static final int CLOUD_HOST_RAM = 8192;  // MB
    private static final int CLOUD_HOST_BW = 10000;  // Mbps
    private static final int CLOUD_HOST_STORAGE = 1000000;  // MB
    private static final int CLOUD_HOST_PES = 8;

    // Cloudlet configuration
    private static final int LOCAL_TRAINING_LENGTH = 10000;  // MI
    private static final int AGGREGATION_LENGTH = 5000;      // MI
    private static final int CLOUDLET_PES = 1;

    private final CloudSimPlus simulation;
    private DatacenterBroker edgeBroker;
    private DatacenterBroker cloudBroker;
    private List<Vm> edgeVmList;
    private List<Vm> cloudVmList;
    private List<Cloudlet> edgeCloudletList;
    private List<Cloudlet> cloudCloudletList;

    public static void main(String[] args) {
        new TrafficEdgeCloudSimulation();
    }

    public TrafficEdgeCloudSimulation() {
        System.out.println("=".repeat(60));
        System.out.println("Traffic Signal Control - Edge/Cloud Simulation");
        System.out.println("=".repeat(60));

        simulation = new CloudSimPlus();

        // Create edge and cloud infrastructure
        Datacenter edgeDatacenter = createEdgeDatacenter();
        Datacenter cloudDatacenter = createCloudDatacenter();

        // Create brokers
        edgeBroker = new DatacenterBrokerSimple(simulation);
        cloudBroker = new DatacenterBrokerSimple(simulation);

        // Create VMs
        edgeVmList = createEdgeVms();
        cloudVmList = createCloudVms();

        // Create cloudlets (tasks)
        edgeCloudletList = createLocalTrainingCloudlets();
        cloudCloudletList = createAggregationCloudlets();

        // Submit to brokers
        edgeBroker.submitVmList(edgeVmList);
        edgeBroker.submitCloudletList(edgeCloudletList);

        cloudBroker.submitVmList(cloudVmList);
        cloudBroker.submitCloudletList(cloudCloudletList);

        // Run simulation
        simulation.start();

        // Print results
        printResults();
    }

    /**
     * Creates the edge datacenter representing traffic intersection infrastructure.
     */
    private Datacenter createEdgeDatacenter() {
        List<Host> hostList = new ArrayList<>();

        for (int i = 0; i < NUM_EDGE_NODES; i++) {
            List<Pe> peList = new ArrayList<>();
            peList.add(new PeSimple(EDGE_VM_MIPS));

            Host host = new HostSimple(EDGE_VM_RAM, EDGE_VM_BW, EDGE_VM_SIZE, peList);
            hostList.add(host);
        }

        Datacenter dc = new DatacenterSimple(simulation, hostList);
        dc.setName("EdgeDatacenter");
        return dc;
    }

    /**
     * Creates the cloud datacenter for federated aggregation.
     */
    private Datacenter createCloudDatacenter() {
        List<Host> hostList = new ArrayList<>();

        for (int i = 0; i < CLOUD_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < CLOUD_HOST_PES; j++) {
                peList.add(new PeSimple(CLOUD_HOST_MIPS));
            }

            Host host = new HostSimple(CLOUD_HOST_RAM, CLOUD_HOST_BW, CLOUD_HOST_STORAGE, peList);
            hostList.add(host);
        }

        Datacenter dc = new DatacenterSimple(simulation, hostList);
        dc.setName("CloudDatacenter");
        return dc;
    }

    /**
     * Creates VMs for edge nodes (traffic intersections).
     */
    private List<Vm> createEdgeVms() {
        List<Vm> vmList = new ArrayList<>();

        for (int i = 0; i < NUM_EDGE_NODES; i++) {
            Vm vm = new VmSimple(EDGE_VM_MIPS, 1);
            vm.setRam(EDGE_VM_RAM);
            vm.setBw(EDGE_VM_BW);
            vm.setSize(EDGE_VM_SIZE);
            vm.setDescription("EdgeNode_" + i);
            vmList.add(vm);
        }

        return vmList;
    }

    /**
     * Creates VMs for cloud aggregation.
     */
    private List<Vm> createCloudVms() {
        List<Vm> vmList = new ArrayList<>();

        // One cloud VM for aggregation
        Vm vm = new VmSimple(CLOUD_HOST_MIPS, CLOUD_HOST_PES);
        vm.setRam(CLOUD_HOST_RAM);
        vm.setBw(CLOUD_HOST_BW);
        vm.setSize(CLOUD_HOST_STORAGE);
        vm.setDescription("CloudAggregator");
        vmList.add(vm);

        return vmList;
    }

    /**
     * Creates cloudlets for local ML training at edge nodes.
     */
    private List<Cloudlet> createLocalTrainingCloudlets() {
        List<Cloudlet> cloudletList = new ArrayList<>();

        for (int i = 0; i < NUM_EDGE_NODES; i++) {
            // Simulate multiple training rounds
            for (int round = 0; round < 5; round++) {
                Cloudlet cloudlet = new CloudletSimple(LOCAL_TRAINING_LENGTH, CLOUDLET_PES);
                cloudlet.setUtilizationModelCpu(new UtilizationModelDynamic(0.8));
                cloudlet.setUtilizationModelRam(new UtilizationModelDynamic(0.5));
                cloudletList.add(cloudlet);
            }
        }

        return cloudletList;
    }

    /**
     * Creates cloudlets for global model aggregation.
     */
    private List<Cloudlet> createAggregationCloudlets() {
        List<Cloudlet> cloudletList = new ArrayList<>();

        // Simulate aggregation rounds
        for (int round = 0; round < 5; round++) {
            Cloudlet cloudlet = new CloudletSimple(AGGREGATION_LENGTH, CLOUDLET_PES);
            cloudlet.setUtilizationModelCpu(new UtilizationModelDynamic(0.9));
            cloudlet.setUtilizationModelRam(new UtilizationModelDynamic(0.7));
            cloudletList.add(cloudlet);
        }

        return cloudletList;
    }

    /**
     * Prints simulation results.
     */
    private void printResults() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("SIMULATION RESULTS");
        System.out.println("=".repeat(60));

        // Edge node results
        System.out.println("\n--- Edge Node (Local Training) Results ---");
        System.out.printf("%-10s %-10s %-15s %-15s %-15s%n",
                "Cloudlet", "VM", "Start Time", "Finish Time", "Exec Time");
        System.out.println("-".repeat(65));

        double totalEdgeExecTime = 0;
        List<Cloudlet> finishedEdge = edgeBroker.getCloudletFinishedList();
        for (Cloudlet cloudlet : finishedEdge) {
            double execTime = cloudlet.getTotalExecutionTime();
            totalEdgeExecTime += execTime;
            System.out.printf("%-10d %-10d %-15.2f %-15.2f %-15.2f%n",
                    cloudlet.getId(),
                    cloudlet.getVm().getId(),
                    cloudlet.getStartTime(),
                    cloudlet.getFinishTime(),
                    execTime);
        }

        // Cloud aggregation results
        System.out.println("\n--- Cloud (Aggregation) Results ---");
        System.out.printf("%-10s %-10s %-15s %-15s %-15s%n",
                "Cloudlet", "VM", "Start Time", "Finish Time", "Exec Time");
        System.out.println("-".repeat(65));

        double totalCloudExecTime = 0;
        List<Cloudlet> finishedCloud = cloudBroker.getCloudletFinishedList();
        for (Cloudlet cloudlet : finishedCloud) {
            double execTime = cloudlet.getTotalExecutionTime();
            totalCloudExecTime += execTime;
            System.out.printf("%-10d %-10d %-15.2f %-15.2f %-15.2f%n",
                    cloudlet.getId(),
                    cloudlet.getVm().getId(),
                    cloudlet.getStartTime(),
                    cloudlet.getFinishTime(),
                    execTime);
        }

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("SUMMARY");
        System.out.println("=".repeat(60));
        System.out.printf("Total Edge Cloudlets: %d%n", finishedEdge.size());
        System.out.printf("Total Cloud Cloudlets: %d%n", finishedCloud.size());
        System.out.printf("Total Edge Execution Time: %.2f seconds%n", totalEdgeExecTime);
        System.out.printf("Total Cloud Execution Time: %.2f seconds%n", totalCloudExecTime);
        System.out.printf("Average Edge Task Time: %.2f seconds%n",
                finishedEdge.isEmpty() ? 0 : totalEdgeExecTime / finishedEdge.size());
        System.out.printf("Average Cloud Task Time: %.2f seconds%n",
                finishedCloud.isEmpty() ? 0 : totalCloudExecTime / finishedCloud.size());
        System.out.println("=".repeat(60));
    }
}
