def _write_selected_variables_to_excel(self, model,
                                       filepath: str,
                                       meta: dict | None = None) -> None:
    """
    Export selected variables and key parameters to a multi-sheet .xlsx workbook.
    """
    import pyomo.environ as pyo
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Prepare some utility functions
    def _pyo_val(x):
        try:
            return float(pyo.value(x))
        except Exception:
            return None

    def _var_to_dataframe(var, value_name="value", col_prefix="i"):
        """
        Serializes a (possibly indexed) Pyomo Var to a flat DataFrame.
        - For scalar vars, returns one-row df with the value.
        - For indexed vars, each index component becomes a separate column i0,i1,...
        """
        rows = []
        if var.is_indexed():
            for idx in var:
                val = pyo.value(var[idx])
                if val is None:
                    continue
                # idx may be scalar or tuple
                if not isinstance(idx, tuple):
                    idx = (idx,)
                row = {f"{col_prefix}{k}": v for k, v in enumerate(idx)}
                row[value_name] = float(val)
                rows.append(row)
        else:
            val = pyo.value(var)
            if val is not None:
                rows.append({value_name: float(val)})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[value_name])

    def _sum_product(var, *possible_params):
        """
        Flexible fallback: sum( param[idx] * var[idx] ) over var's support.
        Tries provided parameter names in order; uses the first one that exists and is indexed compatibly.
        If none found or shapes don’t align, returns None.
        """
        # pick the first available param
        chosen = None
        for pname in possible_params:
            if hasattr(model, pname):
                chosen = getattr(model, pname)
                break
        if chosen is None:
            return None

        total = 0.0
        try:
            if var.is_indexed():
                for idx in var:
                    v = pyo.value(var[idx])
                    if v is None or abs(v) == 0:
                        continue
                    # chosen may be scalar (same for all) or indexed
                    if chosen.is_indexed():
                        try:
                            c = pyo.value(chosen[idx])
                        except Exception:
                            # If index shapes differ, try simple lookup when idx is single and chosen has 1D
                            if not isinstance(idx, tuple):
                                c = pyo.value(chosen[idx])
                            else:
                                # Give up if clearly incompatible
                                return None
                    else:
                        c = pyo.value(chosen)
                    if c is None:
                        continue
                    total += float(c) * float(v)
            else:
                v = pyo.value(var)
                if v is None:
                    return None
                c = pyo.value(chosen) if not chosen.is_indexed() else None
                if c is None:
                    return None
                total = float(c) * float(v)
        except Exception:
            return None
        return total

    # 2) Build the meta as default if it is not given
    if meta is None:
        # meta = {
        #     "written_at": datetime.now().isoformat(timespec="seconds"),
        #     "network_name": self.network_name,
        #     "windstorm_name": self.windstorm_name,
        #     "windstorm_random_seed": getattr(self.meta, 'ws_seed', None),
        #     "number_of_ws_scenarios": getattr(self.meta, 'n_ws_scenarios', 0),
        #     "normal_scenario_included": getattr(self.meta, 'normal_scenario_included', False),
        #     "normal_scenario_probability": getattr(self.meta, 'normal_scenario_prob', 0),
        #     "resilience_metric_threshold": float(pyo.value(model.resilience_metric_threshold)),
        #     "objective_value": float(pyo.value(model.Objective)),
        #     "total_investment_cost": float(pyo.value(model.total_inv_cost_expr)),
        #     "expected_total_operational_cost": float(pyo.value(model.exp_total_op_cost_expr)),
        #     "ws_expected_total_operational_cost_dn": float(pyo.value(model.exp_total_op_cost_dn_ws_expr)),
        #     "ws_exp_total_eens_dn": float(pyo.value(model.exp_total_eens_dn_ws_expr)),
        # }
        #
        # # Add normal scenario costs if included
        # if hasattr(model, 'normal_cost_contribution'):
        #     meta["normal_operation_cost_contribution"] = float(pyo.value(model.normal_cost_contribution))
        #
        #     # Also add the DN generation cost from normal scenario if available
        #     if hasattr(model, 'normal_gen_cost_dn'):
        #         meta["normal_operation_gen_cost_dn"] = float(pyo.value(model.normal_gen_cost_dn))
        #
        #     # Calculate total expected cost including normal
        #     # The windstorm operational cost is already weighted by (1 - normal_prob) in the objective
        #     # So the total is just the objective value
        #     meta["total_expected_operational_cost_with_normal"] = (
        #             float(pyo.value(model.exp_total_op_cost_expr)) +
        #             float(pyo.value(model.normal_cost_contribution))
        #     )
        #
        # # Add detailed normal operation info if available
        # if hasattr(self.meta, 'normal_operation_opf_results') and self.meta.normal_operation_opf_results:
        #     meta["normal_hours_computed"] = self.meta.normal_operation_opf_results.get("hours_computed", "N/A")
        #     meta["normal_scale_factor"] = self.meta.normal_operation_opf_results.get("scale_factor", "N/A")
        #     meta["normal_solver_status"] = self.meta.normal_operation_opf_results.get("solver_status", "N/A")
        #     if hasattr(self.meta.normal_operation_opf_results, 'representative_days'):
        #         meta["normal_representative_days"] = str(
        #             self.meta.normal_operation_opf_results.get("representative_days", "N/A"))

        meta = {
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }

        if hasattr(model, "Objective"):
            meta["objective_value"] = _pyo_val(model.Objective)
        if hasattr(model, "total_inv_cost_expr"):
            meta["total_investment_cost"] = _pyo_val(model.total_inv_cost_expr)
        if hasattr(model, "exp_total_op_cost_expr"):
            meta["expected_total_operational_cost"] = _pyo_val(model.exp_total_op_cost_expr)

        if hasattr(self, "network_name"):
            meta["network_name"] = self.network_name
        if hasattr(self, "windstorm_name"):
            meta["windstorm_name"] = self.windstorm_name
        if hasattr(self, "meta"):
            # Try to expose common run metadata if present on self.meta
            for k in ("ws_seed", "n_ws_scenarios"):
                if hasattr(self.meta, k):
                    meta_key = {
                        "ws_seed": "windstorm_random_seed",
                        "n_ws_scenarios": "number_of_ws_scenarios",
                    }[k]
                    meta[meta_key] = getattr(self.meta, k)

        # 2.1) Line hardening
        if hasattr(model, "total_inv_cost_line_hrdn_expr"):
            meta["investment_cost_line_hardening"] = _pyo_val(model.total_inv_cost_line_hrdn_expr)
        else:
            # Fallback: sum(line_hrdn_cost[l] * line_hrdn[l])
            if hasattr(model, "line_hrdn"):
                line_hrdn_cost = _sum_product(
                    getattr(model, "line_hrdn"),
                    "line_hrdn_cost", "inv_cost_line_hrdn", "line_hardening_cost"
                )
                if line_hrdn_cost is not None:
                    meta["investment_cost_line_hardening"] = float(line_hrdn_cost)
                # else: leave absent if we cannot infer

        # 2.2) DG installation
        if hasattr(model, "total_inv_cost_dg_expr"):
            meta["investment_cost_dg"] = _pyo_val(model.total_inv_cost_dg_expr)
        else:
            # Fallback: sum(dg_inv_cost[idx] * dg_install_capacity[idx])
            if hasattr(model, "dg_install_capacity"):
                dg_cost = _sum_product(
                    getattr(model, "dg_install_capacity"),
                    # Try common naming patterns
                    "dg_inv_cost", "dg_capex", "dg_cost", "dg_unit_cost"
                )
                if dg_cost is not None:
                    meta["investment_cost_dg"] = float(dg_cost)

        # 2.3) ESS installation
        if hasattr(model, "total_inv_cost_ess_expr"):
            meta["investment_cost_ess"] = _pyo_val(model.total_inv_cost_ess_expr)
        else:
            # Fallbacks:
            # (a) One-part cost: sum(ess_inv_cost[idx] * ess_install_capacity[idx])
            # (b) Two-part cost: power + energy components if you model both
            ess_cost_total = None
            if hasattr(model, "ess_install_capacity"):
                one_part = _sum_product(
                    getattr(model, "ess_install_capacity"),
                    "ess_inv_cost", "ess_capex", "ess_cost", "ess_unit_cost"
                )
                if one_part is not None:
                    ess_cost_total = float(one_part)
            # Optional: separate power/energy sizing (only if such vars/params exist)
            # This adds up if it can find both parts; otherwise it leaves the existing total.
            if hasattr(model, "ess_power_install") and hasattr(model, "ess_power_inv_cost"):
                part = _sum_product(getattr(model, "ess_power_install"), "ess_power_inv_cost")
                if part is not None:
                    ess_cost_total = (ess_cost_total or 0.0) + float(part)
            if hasattr(model, "ess_energy_install") and hasattr(model, "ess_energy_inv_cost"):
                part = _sum_product(getattr(model, "ess_energy_install"), "ess_energy_inv_cost")
                if part is not None:
                    ess_cost_total = (ess_cost_total or 0.0) + float(part)
            if ess_cost_total is not None:
                meta["investment_cost_ess"] = ess_cost_total

    # 3) Write results
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:

        # Existing: line hardening sheet (only if variable exists)
        if hasattr(model, "line_hrdn"):
            df_line = _var_to_dataframe(getattr(model, "line_hrdn"), value_name="line_hrdn")
            # Optional: if you carry a mapping from line index to line name, you can left-merge here.

            df_line.to_excel(writer, sheet_name="line_hrdn", index=False)

        # NEW: DG installation capacity
        if hasattr(model, "dg_install_capacity"):
            df_dg = _var_to_dataframe(getattr(model, "dg_install_capacity"),
                                      value_name="dg_install_capacity")
            # TODO: if you have (bus, tech) indices and want named columns, rename i0->bus, i1->tech here:
            # df_dg.rename(columns={"i0": "bus", "i1": "tech"}, inplace=True)
            df_dg.to_excel(writer, sheet_name="dg_install", index=False)

        # NEW: ESS installation capacity
        if hasattr(model, "ess_install_capacity"):
            df_ess = _var_to_dataframe(getattr(model, "ess_install_capacity"),
                                       value_name="ess_install_capacity")
            # TODO: similarly, if indexed by (bus) or (bus, block), rename i* to semantic names:
            # df_ess.rename(columns={"i0": "bus"}, inplace=True)
            df_ess.to_excel(writer, sheet_name="ess_install", index=False)

        # Meta sheet at the end
        pd.DataFrame.from_dict(meta, orient="index", columns=["value"]).to_excel(
            writer, sheet_name="Meta"
        )