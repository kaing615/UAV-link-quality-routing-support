import http from "k6/http";
import { sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 10 },
    { duration: "30s", target: 20 },
    { duration: "20s", target: 5 },
  ],
};

export default function () {
  const payload = {
    nodes: [
      {
        node_id: 1,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        vx: 0.0,
        vy: 0.0,
        vz: 0.0,
        speed: 0.0,
        degree: 1,
      },
      {
        node_id: 2,
        x: 1.0,
        y: 0.0,
        z: 0.0,
        vx: 0.0,
        vy: 0.0,
        vz: 0.0,
        speed: 0.0,
        degree: 1,
      },
    ],
    edges: [
      {
        src: 1,
        dst: 2,
        distance: 10.0,
        rssi: -60.0,
        snr: 20.0,
        delay: 1.0,
        packet_loss: 0.0,
        relative_speed: 0.0,
        throughput: 100.0,
      },
    ],
    query_edges: [[1, 2]],
  };

  http.post(
    "https://api-staging.example.com/predict",
    JSON.stringify(payload),
    { headers: { "Content-Type": "application/json" } },
  );
  sleep(1);
}
