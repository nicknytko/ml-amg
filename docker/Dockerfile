FROM firedrakeproject/firedrake
USER root
RUN apt update && apt upgrade -y
RUN apt install -y zsh
USER firedrake
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN pip3 install pyamg
COPY start.sh /home/firedrake/start.sh
RUN echo "source start.sh" >> /home/firedrake/.zshrc
CMD ["/bin/zsh"]