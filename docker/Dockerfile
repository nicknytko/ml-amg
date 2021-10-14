FROM firedrakeproject/firedrake
USER root
RUN apt update && apt upgrade -y
RUN apt install -y zsh
RUN /home/firedrake/firedrake/bin/pip3 install pyamg torch
RUN /home/firedrake/firedrake/bin/pip3 install torch_geometric torch_sparse torch_scatter
USER firedrake
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
COPY start.sh /home/firedrake/start.sh
RUN echo "source start.sh" >> /home/firedrake/.zshrc
CMD ["/bin/zsh"]