time = model_sol.time  #solution.t#['t_arr']
model_sol_y = model_sol.state_variables['v']  #model_sol.solution.y[:,0]
model_forcing = model_sol.diagnostic_variables['insolation']
print(len(time), len(model_forcing), len(model_sol_y))

df = pd.DataFrame({'time':time+length, 'ice_volume': model_sol_y, 'insolation':model_forcing})