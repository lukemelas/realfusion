import { Stack, StackProps } from '@chakra-ui/react'
import { Container } from 'next/app'

export const Main = (props: StackProps) => (
  <Container
    spacing="1.5rem"
    width="100%"
    maxWidth="52rem"
    mt="-45vh"
    pt="8rem"
    px="1rem"
    {...props}
  >
    {props.children}
  </Container>
)
