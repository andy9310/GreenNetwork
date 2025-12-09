def _get_obs_for_node(self, i: int) -> np.ndarray:
        neighbors = sorted(list(self.graph.neighbors(i)))
        deg_i = len(neighbors)
        deg_norm = deg_i / max(self.max_degree, 1)

        mode_i = self.node_modes.get(i, 1)
        mode_norm = mode_i / 2.0  # 0,0.5,1

        # Compute local aggregate load
        local_load = 0.0
        for j in neighbors:
            key = self._edge_key(i, j)
            local_load += self.edge_load.get(key, 0.0)
        # normalize by base_capacity * max_degree
        denom = self.base_capacity * max(self.max_degree, 1)
        local_load_norm = local_load / denom if denom > 0 else 0.0

        # per-edge features: (util, overload_flag) for up to max_degree neighbors
        edge_feats = np.zeros(2 * self.max_degree, dtype=np.float32)
        for idx, j in enumerate(neighbors):
            if idx >= self.max_degree:
                break
            key = self._edge_key(i, j)
            load = self.edge_load.get(key, 0.0)
            capacity = self.edge_capacity.get(key, self.base_capacity)
            util = load / capacity if capacity > 0 else 0.0
            util_clipped = float(min(util, 2.0))  # clip for stability
            overload_flag = 1.0 if util > 1.0 else 0.0

            edge_feats[2 * idx] = util_clipped
            edge_feats[2 * idx + 1] = overload_flag

        obs = np.concatenate(
            [
                np.array([deg_norm, mode_norm, local_load_norm], dtype=np.float32),
                edge_feats,
            ]
        )
        assert obs.shape[0] == self.obs_dim
        return obs

def _simulate_traffic_step(self):
        """
        Generate random flows, route them on shortest paths,
        compute link loads, utilization, energy, delay, overload.
        """
        # reset loads
        for key in self.edge_load.keys():
            self.edge_load[key] = 0.0

        # update capacities based on node modes
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            mode_u = self.node_modes[u]
            mode_v = self.node_modes[v]
            factor_u = self.mode_capacity_factor[mode_u]
            factor_v = self.mode_capacity_factor[mode_v]
            # use min of both ends to be conservative
            cap_factor = min(factor_u, factor_v)
            self.edge_capacity[key] = self.base_capacity * cap_factor

        # generate flows: random src/dst pairs with random demand
        flows = []
        for _ in range(self.num_flows_per_step):
            s = self.rng.choice(self.nodes)
            d = self.rng.choice(self.nodes)
            while d == s:
                d = self.rng.choice(self.nodes)
            # demand: positive, around base_capacity / 4
            demand = max(
                0.1,
                float(
                    self.np_rng.normal(self.base_capacity / 4.0,
                                       self.base_capacity / 10.0)
                ),
            )
            flows.append((s, d, demand))

        # route flows using shortest paths by hop count
        disconnect_penalty = 0.0
        for s, d, demand in flows:
            try:
                path = nx.shortest_path(self.graph, s, d)
            except nx.NetworkXNoPath:
                # if disconnected, add penalty
                disconnect_penalty += demand * 10.0
                continue

            # accumulate load on each edge along the path
            for u, v in zip(path[:-1], path[1:]):
                key = self._edge_key(u, v)
                self.edge_load[key] += demand

        total_energy = 0.0
        total_delay = 0.0
        total_overload = disconnect_penalty

        # compute per-edge metrics
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            load = self.edge_load[key]
            capacity = self.edge_capacity[key]
            if capacity <= 0.0:
                # if capacity is zero, treat as heavy overload
                util = 10.0
            else:
                util = load / capacity

            # overload when util > 1
            overload = max(0.0, util - 1.0)
            # simple convex delay: base_delay * (1 + util^2)
            delay = self.base_delay * (1.0 + util * util)

            mode_u = self.node_modes[u]
            mode_v = self.node_modes[v]
            power_factor = 0.5 * (
                self.mode_power_factor[mode_u] + self.mode_power_factor[mode_v]
            )
            # energy = base_power * factor * (1 + 0.5 * util)
            energy = self.base_power * power_factor * (1.0 + 0.5 * util)

            total_energy += energy
            total_delay += delay
            total_overload += overload * 5.0  # scale up overload cost
        return total_energy, total_delay, total_overload

    def _get_obs_all(self) -> Dict[int, np.ndarray]:
        return {i: self._get_obs_for_node(i) for i in self.nodes}