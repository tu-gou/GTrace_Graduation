import torch
import torch.nn as nn
import dgl


class GraphLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, graph, input_data, hidden_state):
        if hidden_state is None:
            hx = [torch.zeros(graph.number_of_nodes(), self.hidden_size).to(input_data.device),
                  torch.zeros(graph.number_of_nodes(), self.hidden_size).to(input_data.device)]
        else:
            hx = hidden_state

        # input_concat = torch.cat([input_data, hx[0]], dim=-1)
        input_concat = torch.cat([input_data, hx[0]], dim=-1)
        graph.ndata['h_concat'] = input_concat

        graph.ndata['iouf'] = self.graph_linear(graph.ndata['h_concat'])

        # LSTM computation
        i, o, u, f = torch.split(graph.ndata['iouf'], self.hidden_size, dim=-1)
        i, o, u, f = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u), torch.sigmoid(f)

        cy = (i * u) + (f * hx[1])
        hy = o * torch.tanh(cy)

        return hy, cy


class GraphLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphLSTM, self).__init__()
        self.cell = GraphLSTMCell(input_size, input_size)
        self.linear_o = nn.Linear(input_size, output_size)

    def forward(self, graph, input_data: torch.Tensor):
        hidden_state = None
        hidden_states = []
        cell_states = []

        for step in range(input_data.size(1)):
            input_step = input_data[:, step, :]

            # hidden_state, cell_state = self.cell(graph, input_step, hidden_state)
            hidden_state, cell_state = self.cell(graph, input_step, None)

            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

        # Get the last hidden state and apply linear transformation
        output = self.linear_o(hidden_states[-1])

        return output

