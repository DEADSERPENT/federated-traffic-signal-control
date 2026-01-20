/*
 * NS-3 Simulation Module for Federated Learning Traffic Signal Control
 *
 * This module simulates realistic V2I (Vehicle-to-Infrastructure) communication
 * for FL-based traffic signal optimization.
 *
 * Features:
 * - 802.11p (DSRC) communication for V2I
 * - Realistic urban traffic topology
 * - FL model update transmission simulation
 * - Network metrics collection (latency, packet loss, throughput)
 *
 * Author: FL Traffic Signal Control Project
 *
 * Build: ./ns3 run scratch/fl-traffic/fl_traffic_network
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wave-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/stats-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLTrafficNetwork");

//==============================================================================
// Global Statistics Collector
//==============================================================================
struct NetworkStats {
    uint64_t totalBytesSent = 0;
    uint64_t totalBytesReceived = 0;
    uint64_t totalPacketsSent = 0;
    uint64_t totalPacketsReceived = 0;
    uint64_t totalPacketsLost = 0;
    double totalDelay = 0.0;
    uint32_t delayCount = 0;

    std::vector<double> latencies;
    std::vector<double> throughputs;

    double GetAverageLatency() const {
        return delayCount > 0 ? totalDelay / delayCount : 0.0;
    }

    double GetPacketLossRate() const {
        return totalPacketsSent > 0 ?
            (double)totalPacketsLost / totalPacketsSent : 0.0;
    }

    double GetThroughput() const {
        return totalBytesReceived * 8.0; // bits
    }
};

static NetworkStats g_stats;
static std::map<uint64_t, Time> g_packetSendTime;

//==============================================================================
// Packet Trace Callbacks
//==============================================================================
void PacketSentCallback(Ptr<const Packet> packet) {
    g_stats.totalPacketsSent++;
    g_stats.totalBytesSent += packet->GetSize();
    g_packetSendTime[packet->GetUid()] = Simulator::Now();
}

void PacketReceivedCallback(Ptr<const Packet> packet) {
    g_stats.totalPacketsReceived++;
    g_stats.totalBytesReceived += packet->GetSize();

    auto it = g_packetSendTime.find(packet->GetUid());
    if (it != g_packetSendTime.end()) {
        double delay = (Simulator::Now() - it->second).GetMilliSeconds();
        g_stats.totalDelay += delay;
        g_stats.delayCount++;
        g_stats.latencies.push_back(delay);
        g_packetSendTime.erase(it);
    }
}

//==============================================================================
// FL Model Update Application
//==============================================================================
class FLUpdateApplication : public Application {
public:
    FLUpdateApplication();
    virtual ~FLUpdateApplication();

    static TypeId GetTypeId();

    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
               uint32_t nPackets, DataRate dataRate, uint32_t clientId);

private:
    virtual void StartApplication();
    virtual void StopApplication();

    void ScheduleTx();
    void SendPacket();

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    uint32_t m_nPackets;
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running;
    uint32_t m_packetsSent;
    uint32_t m_clientId;
};

FLUpdateApplication::FLUpdateApplication()
    : m_socket(nullptr), m_packetSize(0), m_nPackets(0),
      m_dataRate(0), m_running(false), m_packetsSent(0), m_clientId(0) {}

FLUpdateApplication::~FLUpdateApplication() {
    m_socket = nullptr;
}

TypeId FLUpdateApplication::GetTypeId() {
    static TypeId tid = TypeId("FLUpdateApplication")
        .SetParent<Application>()
        .SetGroupName("Applications")
        .AddConstructor<FLUpdateApplication>();
    return tid;
}

void FLUpdateApplication::Setup(Ptr<Socket> socket, Address address,
                                  uint32_t packetSize, uint32_t nPackets,
                                  DataRate dataRate, uint32_t clientId) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_clientId = clientId;
}

void FLUpdateApplication::StartApplication() {
    m_running = true;
    m_packetsSent = 0;
    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void FLUpdateApplication::StopApplication() {
    m_running = false;
    if (m_sendEvent.IsPending()) {
        Simulator::Cancel(m_sendEvent);
    }
    if (m_socket) {
        m_socket->Close();
    }
}

void FLUpdateApplication::SendPacket() {
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    m_socket->Send(packet);
    PacketSentCallback(packet);

    NS_LOG_INFO("Client " << m_clientId << " sent FL update packet "
                << m_packetsSent << " at " << Simulator::Now().GetSeconds() << "s");

    if (++m_packetsSent < m_nPackets) {
        ScheduleTx();
    }
}

void FLUpdateApplication::ScheduleTx() {
    if (m_running) {
        Time tNext(Seconds(m_packetSize * 8 /
                          static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &FLUpdateApplication::SendPacket, this);
    }
}

//==============================================================================
// Packet Sink for Server
//==============================================================================
class FLServerSink : public Application {
public:
    FLServerSink();
    virtual ~FLServerSink();

    static TypeId GetTypeId();
    uint64_t GetTotalRx() const { return m_totalRx; }

private:
    virtual void StartApplication();
    virtual void StopApplication();

    void HandleRead(Ptr<Socket> socket);

    Ptr<Socket> m_socket;
    uint64_t m_totalRx;
    Address m_local;
};

FLServerSink::FLServerSink() : m_socket(nullptr), m_totalRx(0) {}
FLServerSink::~FLServerSink() { m_socket = nullptr; }

TypeId FLServerSink::GetTypeId() {
    static TypeId tid = TypeId("FLServerSink")
        .SetParent<Application>()
        .SetGroupName("Applications")
        .AddConstructor<FLServerSink>();
    return tid;
}

void FLServerSink::StartApplication() {
    if (!m_socket) {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), 9999);
        m_socket->Bind(local);
    }
    m_socket->SetRecvCallback(MakeCallback(&FLServerSink::HandleRead, this));
}

void FLServerSink::StopApplication() {
    if (m_socket) {
        m_socket->Close();
        m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    }
}

void FLServerSink::HandleRead(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from))) {
        m_totalRx += packet->GetSize();
        PacketReceivedCallback(packet);

        NS_LOG_INFO("Server received " << packet->GetSize() << " bytes at "
                    << Simulator::Now().GetSeconds() << "s");
    }
}

//==============================================================================
// Main FL Traffic Network Simulation
//==============================================================================
class FLTrafficNetworkSimulation {
public:
    FLTrafficNetworkSimulation(uint32_t numIntersections, uint32_t numVehicles);

    void Configure();
    void Run(double duration);
    void PrintResults(const std::string& outputFile);

private:
    void CreateNodes();
    void ConfigureWifi();
    void ConfigureMobility();
    void ConfigureInternet();
    void ConfigureApplications();
    void InstallFlowMonitor();

    uint32_t m_numIntersections;
    uint32_t m_numVehicles;
    double m_simDuration;

    NodeContainer m_intersectionNodes;  // RSUs at intersections
    NodeContainer m_vehicleNodes;        // Vehicles
    NodeContainer m_serverNode;          // Central FL server

    NetDeviceContainer m_intersectionDevices;
    NetDeviceContainer m_vehicleDevices;
    NetDeviceContainer m_serverDevices;

    Ipv4InterfaceContainer m_intersectionInterfaces;
    Ipv4InterfaceContainer m_vehicleInterfaces;
    Ipv4InterfaceContainer m_serverInterfaces;

    Ptr<FlowMonitor> m_flowMonitor;
};

FLTrafficNetworkSimulation::FLTrafficNetworkSimulation(uint32_t numIntersections,
                                                         uint32_t numVehicles)
    : m_numIntersections(numIntersections),
      m_numVehicles(numVehicles),
      m_simDuration(100.0) {}

void FLTrafficNetworkSimulation::CreateNodes() {
    NS_LOG_INFO("Creating nodes...");

    // Create intersection RSU nodes
    m_intersectionNodes.Create(m_numIntersections);

    // Create vehicle nodes
    m_vehicleNodes.Create(m_numVehicles);

    // Create central server node
    m_serverNode.Create(1);

    NS_LOG_INFO("Created " << m_numIntersections << " intersections, "
                << m_numVehicles << " vehicles, 1 server");
}

void FLTrafficNetworkSimulation::ConfigureWifi() {
    NS_LOG_INFO("Configuring 802.11p (WAVE/DSRC)...");

    // WAVE PHY and Channel
    YansWifiChannelHelper waveChannel = YansWifiChannelHelper::Default();
    waveChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    waveChannel.AddPropagationLoss("ns3::FriisPropagationLossModel",
                                    "Frequency", DoubleValue(5.9e9));

    YansWifiPhyHelper wavePhy;
    wavePhy.SetChannel(waveChannel.Create());
    wavePhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    wavePhy.Set("TxPowerStart", DoubleValue(20.0));
    wavePhy.Set("TxPowerEnd", DoubleValue(20.0));

    // WAVE MAC - 802.11p
    QosWaveMacHelper waveMac = QosWaveMacHelper::Default();
    WaveHelper waveHelper = WaveHelper::Default();

    // Install on intersections (RSUs)
    m_intersectionDevices = waveHelper.Install(wavePhy, waveMac, m_intersectionNodes);

    // Install on vehicles
    m_vehicleDevices = waveHelper.Install(wavePhy, waveMac, m_vehicleNodes);

    // Server uses wired connection (simulated as high-speed wifi)
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    YansWifiPhyHelper phy;
    phy.SetChannel(waveChannel.Create());

    m_serverDevices = wifi.Install(phy, mac, m_serverNode);

    NS_LOG_INFO("802.11p configured");
}

void FLTrafficNetworkSimulation::ConfigureMobility() {
    NS_LOG_INFO("Configuring mobility...");

    // Intersections: Fixed positions in a grid
    // Assuming 500m x 500m area with intersections at corners
    MobilityHelper intersectionMobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    double gridSpacing = 500.0;  // meters
    int gridSize = (int)std::ceil(std::sqrt(m_numIntersections));

    for (uint32_t i = 0; i < m_numIntersections; i++) {
        double x = (i % gridSize) * gridSpacing;
        double y = (i / gridSize) * gridSpacing;
        positionAlloc->Add(Vector(x, y, 10.0));  // 10m height for RSU
    }

    intersectionMobility.SetPositionAllocator(positionAlloc);
    intersectionMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    intersectionMobility.Install(m_intersectionNodes);

    // Server: Fixed position at center
    MobilityHelper serverMobility;
    Ptr<ListPositionAllocator> serverPos = CreateObject<ListPositionAllocator>();
    double centerX = (gridSize - 1) * gridSpacing / 2.0;
    double centerY = (gridSize - 1) * gridSpacing / 2.0;
    serverPos->Add(Vector(centerX, centerY, 15.0));
    serverMobility.SetPositionAllocator(serverPos);
    serverMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    serverMobility.Install(m_serverNode);

    // Vehicles: Random waypoint mobility
    MobilityHelper vehicleMobility;
    double maxX = (gridSize - 1) * gridSpacing;
    double maxY = (gridSize - 1) * gridSpacing;

    std::ostringstream xRange, yRange;
    xRange << "ns3::UniformRandomVariable[Min=0|Max=" << maxX << "]";
    yRange << "ns3::UniformRandomVariable[Min=0|Max=" << maxY << "]";

    vehicleMobility.SetPositionAllocator(
        "ns3::RandomBoxPositionAllocator",
        "X", StringValue(xRange.str()),
        "Y", StringValue(yRange.str()),
        "Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"));

    vehicleMobility.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=5|Max=15]"),  // 5-15 m/s
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0]"),
        "PositionAllocator", PointerValue(
            CreateObjectWithAttributes<RandomBoxPositionAllocator>(
                "X", StringValue(xRange.str()),
                "Y", StringValue(yRange.str()),
                "Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"))));

    vehicleMobility.Install(m_vehicleNodes);

    NS_LOG_INFO("Mobility configured");
}

void FLTrafficNetworkSimulation::ConfigureInternet() {
    NS_LOG_INFO("Configuring Internet stack...");

    InternetStackHelper internet;
    internet.Install(m_intersectionNodes);
    internet.Install(m_vehicleNodes);
    internet.Install(m_serverNode);

    Ipv4AddressHelper address;

    // Intersections network
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_intersectionInterfaces = address.Assign(m_intersectionDevices);

    // Vehicles network
    address.SetBase("10.1.2.0", "255.255.255.0");
    m_vehicleInterfaces = address.Assign(m_vehicleDevices);

    // Server network
    address.SetBase("10.1.3.0", "255.255.255.0");
    m_serverInterfaces = address.Assign(m_serverDevices);

    // Enable routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    NS_LOG_INFO("Internet stack configured");
}

void FLTrafficNetworkSimulation::ConfigureApplications() {
    NS_LOG_INFO("Configuring FL applications...");

    // Server application - receives FL updates
    Ptr<FLServerSink> serverApp = CreateObject<FLServerSink>();
    m_serverNode.Get(0)->AddApplication(serverApp);
    serverApp->SetStartTime(Seconds(0.0));
    serverApp->SetStopTime(Seconds(m_simDuration));

    // Each intersection sends FL model updates
    uint16_t serverPort = 9999;
    Address serverAddress(InetSocketAddress(m_serverInterfaces.GetAddress(0), serverPort));

    // FL model update size: ~100KB per update (compressed model parameters)
    uint32_t modelUpdateSize = 100 * 1024;  // 100 KB
    uint32_t numUpdates = 10;  // 10 FL rounds
    DataRate dataRate("6Mbps");  // 802.11p data rate

    for (uint32_t i = 0; i < m_numIntersections; i++) {
        Ptr<Socket> socket = Socket::CreateSocket(
            m_intersectionNodes.Get(i), UdpSocketFactory::GetTypeId());

        Ptr<FLUpdateApplication> app = CreateObject<FLUpdateApplication>();
        app->Setup(socket, serverAddress, modelUpdateSize, numUpdates, dataRate, i);
        m_intersectionNodes.Get(i)->AddApplication(app);

        // Stagger start times to avoid collision
        double startTime = 1.0 + i * 2.0;  // 2 second gap between clients
        app->SetStartTime(Seconds(startTime));
        app->SetStopTime(Seconds(m_simDuration - 1.0));
    }

    NS_LOG_INFO("FL applications configured");
}

void FLTrafficNetworkSimulation::InstallFlowMonitor() {
    FlowMonitorHelper flowHelper;
    m_flowMonitor = flowHelper.InstallAll();
}

void FLTrafficNetworkSimulation::Configure() {
    CreateNodes();
    ConfigureWifi();
    ConfigureMobility();
    ConfigureInternet();
    ConfigureApplications();
    InstallFlowMonitor();
}

void FLTrafficNetworkSimulation::Run(double duration) {
    m_simDuration = duration;

    NS_LOG_INFO("Starting simulation for " << duration << " seconds...");

    Simulator::Stop(Seconds(duration));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete");
}

void FLTrafficNetworkSimulation::PrintResults(const std::string& outputFile) {
    std::cout << "\n============================================================\n";
    std::cout << "  FL TRAFFIC NETWORK SIMULATION RESULTS\n";
    std::cout << "============================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Intersections (RSUs): " << m_numIntersections << "\n";
    std::cout << "  Vehicles: " << m_numVehicles << "\n";
    std::cout << "  Duration: " << m_simDuration << " seconds\n\n";

    std::cout << "Network Statistics:\n";
    std::cout << "  Total Packets Sent: " << g_stats.totalPacketsSent << "\n";
    std::cout << "  Total Packets Received: " << g_stats.totalPacketsReceived << "\n";
    std::cout << "  Packets Lost: " << g_stats.totalPacketsLost << "\n";
    std::cout << "  Packet Loss Rate: " << std::fixed << std::setprecision(4)
              << g_stats.GetPacketLossRate() * 100 << "%\n";
    std::cout << "  Average Latency: " << std::fixed << std::setprecision(2)
              << g_stats.GetAverageLatency() << " ms\n";
    std::cout << "  Total Bytes Transferred: " << g_stats.totalBytesReceived << "\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << g_stats.GetThroughput() / 1e6 << " Mbps\n";

    // Flow monitor statistics
    if (m_flowMonitor) {
        m_flowMonitor->CheckForLostPackets();
        Ptr<Ipv4FlowClassifier> classifier =
            DynamicCast<Ipv4FlowClassifier>(m_flowMonitor->GetClassifier());

        std::map<FlowId, FlowMonitor::FlowStats> stats = m_flowMonitor->GetFlowStats();

        uint64_t totalRxPackets = 0;
        uint64_t totalTxPackets = 0;
        double totalDelaySum = 0;

        for (auto& flow : stats) {
            totalRxPackets += flow.second.rxPackets;
            totalTxPackets += flow.second.txPackets;
            totalDelaySum += flow.second.delaySum.GetSeconds();
        }

        std::cout << "\nFlow Monitor Statistics:\n";
        std::cout << "  Total Flows: " << stats.size() << "\n";
        std::cout << "  Total TX Packets: " << totalTxPackets << "\n";
        std::cout << "  Total RX Packets: " << totalRxPackets << "\n";
    }

    // Write to JSON file
    std::ofstream outFile(outputFile);
    if (outFile.is_open()) {
        outFile << "{\n";
        outFile << "  \"intersections\": " << m_numIntersections << ",\n";
        outFile << "  \"vehicles\": " << m_numVehicles << ",\n";
        outFile << "  \"duration_s\": " << m_simDuration << ",\n";
        outFile << "  \"packets_sent\": " << g_stats.totalPacketsSent << ",\n";
        outFile << "  \"packets_received\": " << g_stats.totalPacketsReceived << ",\n";
        outFile << "  \"packet_loss_rate\": " << g_stats.GetPacketLossRate() << ",\n";
        outFile << "  \"avg_latency_ms\": " << g_stats.GetAverageLatency() << ",\n";
        outFile << "  \"throughput_mbps\": " << g_stats.GetThroughput() / 1e6 << ",\n";
        outFile << "  \"latencies\": [";
        for (size_t i = 0; i < g_stats.latencies.size(); i++) {
            outFile << g_stats.latencies[i];
            if (i < g_stats.latencies.size() - 1) outFile << ", ";
        }
        outFile << "]\n";
        outFile << "}\n";
        outFile.close();

        std::cout << "\nResults saved to: " << outputFile << "\n";
    }

    std::cout << "\n============================================================\n";
}

//==============================================================================
// Main Function
//==============================================================================
int main(int argc, char* argv[]) {
    // Default parameters
    uint32_t numIntersections = 4;
    uint32_t numVehicles = 50;
    double duration = 60.0;
    std::string outputFile = "ns3_results.json";
    bool verbose = false;

    // Command line arguments
    CommandLine cmd;
    cmd.AddValue("intersections", "Number of intersections/RSUs", numIntersections);
    cmd.AddValue("vehicles", "Number of vehicles", numVehicles);
    cmd.AddValue("duration", "Simulation duration (seconds)", duration);
    cmd.AddValue("output", "Output JSON file", outputFile);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.Parse(argc, argv);

    if (verbose) {
        LogComponentEnable("FLTrafficNetwork", LOG_LEVEL_INFO);
    }

    std::cout << "\n============================================================\n";
    std::cout << "  NS-3 FL Traffic Network Simulation\n";
    std::cout << "============================================================\n";
    std::cout << "  Intersections: " << numIntersections << "\n";
    std::cout << "  Vehicles: " << numVehicles << "\n";
    std::cout << "  Duration: " << duration << " seconds\n";
    std::cout << "============================================================\n\n";

    // Create and run simulation
    FLTrafficNetworkSimulation sim(numIntersections, numVehicles);
    sim.Configure();
    sim.Run(duration);
    sim.PrintResults(outputFile);

    return 0;
}
